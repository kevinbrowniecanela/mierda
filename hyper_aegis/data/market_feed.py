"""Feed de mercado asíncrono de Hyperliquid.

- REST (API Info) mediante `hyperliquid-python-sdk`, envuelto en
  `asyncio.to_thread` porque el SDK es síncrono.
- WebSocket nativo (`websockets`) para el libro L2 en tiempo real de la
  watchlist; el último snapshot de cada moneda se mantiene en memoria.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import websockets
from hyperliquid.info import Info

from config.settings import Settings

log = logging.getLogger(__name__)

_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "1d": 86_400_000,
}


@dataclass
class L2Book:
    coin: str
    bids: list[tuple[float, float]] = field(default_factory=list)  # (px, sz)
    asks: list[tuple[float, float]] = field(default_factory=list)
    ts: float = 0.0

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2

    @property
    def spread_bps(self) -> float | None:
        mid = self.mid
        if not mid:
            return None
        return (self.asks[0][0] - self.bids[0][0]) / mid * 10_000

    def imbalance(self, depth_pct: float = 0.005) -> float:
        """Desequilibrio de profundidad en ±depth_pct alrededor del mid.

        Devuelve un valor en [-1, 1]: >0 presión compradora, <0 vendedora.
        """
        mid = self.mid
        if not mid:
            return 0.0
        lo, hi = mid * (1 - depth_pct), mid * (1 + depth_pct)
        bid_ntl = sum(px * sz for px, sz in self.bids if px >= lo)
        ask_ntl = sum(px * sz for px, sz in self.asks if px <= hi)
        total = bid_ntl + ask_ntl
        return 0.0 if total == 0 else (bid_ntl - ask_ntl) / total

    @classmethod
    def from_levels(cls, coin: str, levels: list[list[dict[str, Any]]]) -> "L2Book":
        bids = [(float(lv["px"]), float(lv["sz"])) for lv in levels[0]]
        asks = [(float(lv["px"]), float(lv["sz"])) for lv in levels[1]]
        return cls(coin=coin, bids=bids, asks=asks, ts=time.time())


class MarketFeed:
    """Acceso unificado a datos de mercado de Hyperliquid."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._info: Info | None = None
        self._sem = asyncio.Semaphore(settings.max_concurrent_requests)
        self.books: dict[str, L2Book] = {}
        self._ws_coins: set[str] = set()
        self._ws_task: asyncio.Task | None = None
        self._ws_resubscribe = asyncio.Event()
        self._sz_decimals: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # REST
    # ------------------------------------------------------------------ #
    async def info(self) -> Info:
        if self._info is None:
            self._info = await asyncio.to_thread(Info, self.s.hl_api_url, True)
        return self._info

    async def call(self, fn_name: str, *args: Any, **kwargs: Any) -> Any:
        """Llama a un método del SDK `Info` en un hilo, con límite de concurrencia
        y reintentos exponenciales ante errores de red / rate limit."""
        info = await self.info()
        fn = getattr(info, fn_name)
        delay = 1.0
        for attempt in range(4):
            try:
                async with self._sem:
                    return await asyncio.to_thread(fn, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - el SDK lanza tipos variados
                if attempt == 3:
                    raise
                log.warning("Info.%s falló (%s); reintento en %.0fs", fn_name, exc, delay)
                await asyncio.sleep(delay)
                delay *= 2

    async def meta_and_ctxs(self) -> tuple[list[dict], list[dict]]:
        meta, ctxs = await self.call("meta_and_asset_ctxs")
        universe = meta["universe"]
        for asset in universe:
            self._sz_decimals[asset["name"]] = int(asset["szDecimals"])
        return universe, ctxs

    async def sz_decimals(self, coin: str) -> int:
        if coin not in self._sz_decimals:
            await self.meta_and_ctxs()
        return self._sz_decimals.get(coin, 2)

    async def all_mids(self) -> dict[str, float]:
        mids = await self.call("all_mids")
        return {k: float(v) for k, v in mids.items()}

    async def candles(self, coin: str, interval: str | None = None, lookback: int | None = None) -> pd.DataFrame:
        """Velas OHLCV como DataFrame indexado por tiempo UTC."""
        interval = interval or self.s.candle_interval
        lookback = lookback or self.s.candle_lookback
        end = int(time.time() * 1000)
        start = end - _INTERVAL_MS[interval] * lookback
        raw = await self.call("candles_snapshot", coin, interval, start, end)
        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(raw)
        df = df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "trades"})
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        return df.set_index("time")[["open", "high", "low", "close", "volume"]].sort_index()

    async def l2_snapshot(self, coin: str) -> L2Book:
        """Libro L2: usa el WebSocket si está fresco (<10 s); si no, REST."""
        book = self.books.get(coin)
        if book and time.time() - book.ts < 10:
            return book
        raw = await self.call("l2_snapshot", coin)
        book = L2Book.from_levels(coin, raw["levels"])
        self.books[coin] = book
        return book

    # ------------------------------------------------------------------ #
    # WebSocket L2
    # ------------------------------------------------------------------ #
    def set_ws_coins(self, coins: list[str]) -> None:
        new = set(coins)
        if new != self._ws_coins:
            self._ws_coins = new
            self._ws_resubscribe.set()

    def start_ws(self) -> asyncio.Task:
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_loop(), name="l2-websocket")
        return self._ws_task

    async def _ws_loop(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.s.hl_ws_url, ping_interval=20, max_size=2**23) as ws:
                    log.info("WebSocket L2 conectado (%d monedas)", len(self._ws_coins))
                    backoff = 1.0
                    subscribed: set[str] = set()
                    while True:
                        if self._ws_resubscribe.is_set() or subscribed != self._ws_coins:
                            self._ws_resubscribe.clear()
                            for coin in subscribed - self._ws_coins:
                                await ws.send(json.dumps({"method": "unsubscribe", "subscription": {"type": "l2Book", "coin": coin}}))
                            for coin in self._ws_coins - subscribed:
                                await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}))
                            subscribed = set(self._ws_coins)
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            await ws.send(json.dumps({"method": "ping"}))
                            continue
                        data = json.loads(msg)
                        if data.get("channel") == "l2Book":
                            payload = data["data"]
                            self.books[payload["coin"]] = L2Book.from_levels(payload["coin"], payload["levels"])
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("WebSocket L2 caído (%s); reconexión en %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
