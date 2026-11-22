"""Acciones de EE. UU.: datos, universo, noticias y horario de mercado (Yahoo Finance).

Sustituye a `MarketFeed` + `DerivativesData` cuando `MARKET=stocks`, con la
misma interfaz que usa `run.py`. Solo SIMULACIÓN (paper): no hay broker de
acciones integrado.

- Velas diarias AJUSTADAS por splits/dividendos; solo se devuelven velas
  COMPLETAS (la sesión en curso se descarta).
- Precios en vivo con el endpoint `spark` de Yahoo (retardo típico de 15 min).
- Horario regular NYSE 9:30-16:00 ET, y precios recientes como comprobación
  de festivo (si no hay precios frescos, se considera cerrado).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import aiohttp
import feedparser
import numpy as np
import pandas as pd

from config.settings import Settings
from data.derivatives_data import AssetSnapshot
from data.news_engine import NewsItem

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 HyperAegis-research"}
CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
SPARK = "https://query1.finance.yahoo.com/v8/finance/spark"
RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline"


def ny_now() -> datetime:
    return datetime.now(NY)


def in_regular_hours(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


def parse_daily(result: dict, now: datetime | None = None) -> pd.DataFrame:
    """Velas diarias ajustadas -> DataFrame (índice = fecha UTC). Descarta la vela de hoy si no ha cerrado."""
    now = now or ny_now()
    q = result["indicators"]["quote"][0]
    adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose") or q["close"]
    rows = []
    for i, t in enumerate(result.get("timestamp") or []):
        o, h, l, c, v, a = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i], adj[i]
        if None in (o, h, l, c, a) or not c:
            continue
        k = a / c
        rows.append((t, o * k, h * k, l * k, a, v or 0))
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    a_ = np.array(rows, dtype=float)
    idx = pd.to_datetime(a_[:, 0], unit="s", utc=True).normalize()
    df = pd.DataFrame({"open": a_[:, 1], "high": a_[:, 2], "low": a_[:, 3], "close": a_[:, 4], "volume": a_[:, 5]}, index=idx)
    df = df[~df.index.duplicated(keep="last")]
    # la vela de la sesión en curso (o de hoy antes del cierre) no está completa
    last_day = df.index[-1].date()
    if last_day == now.date() and (now.hour * 60 + now.minute) < 16 * 60 + 10:
        df = df.iloc[:-1]
    return df


class StockFeed:
    """Interfaz compatible con MarketFeed para `run.py`."""

    partial_last_bar = False  # `candles()` ya devuelve solo velas completas

    def __init__(self, settings: Settings, session: aiohttp.ClientSession):
        self.s = settings
        self.http = session
        self.books: dict = {}
        self.coins: set[str] = set()
        self._candles: dict[str, tuple[float, pd.DataFrame]] = {}
        self._mids: dict[str, float] = {}
        self._mids_ts = 0.0
        self.last_quote_ts = 0.0  # hora (epoch) de la última cotización recibida
        self._sem = asyncio.Semaphore(5)

    # --- compatibilidad con MarketFeed ----------------------------------- #
    def set_ws_coins(self, coins: list[str]) -> None:
        self.coins = set(coins) | {self.s.portfolio_core_symbol}  # el núcleo siempre cotiza

    def start_ws(self) -> asyncio.Task:
        async def idle() -> None:
            while True:
                await asyncio.sleep(3600)
        return asyncio.create_task(idle(), name="stocks-idle")

    async def sz_decimals(self, coin: str) -> int:
        return 2  # acciones fraccionadas hasta 0,01

    async def call(self, fn_name: str, *args, **kwargs):
        raise NotImplementedError(f"{fn_name}: no disponible en modo acciones")

    # --- datos ------------------------------------------------------------- #
    async def _get_json(self, url: str, params: dict) -> dict | None:
        delay = 1.0
        for attempt in range(4):
            try:
                async with self._sem, self.http.get(url, params=params, headers=UA, timeout=aiohttp.ClientTimeout(total=25)) as r:
                    if r.status == 200:
                        return await r.json(content_type=None)
                    if r.status == 404:
                        return None
            except Exception as exc:  # noqa: BLE001
                log.debug("Yahoo %s: %s", url, exc)
            await asyncio.sleep(delay)
            delay *= 2
        return None

    async def candles(self, coin: str, interval: str | None = None, lookback: int | None = None) -> pd.DataFrame:
        now = time.time()
        cached = self._candles.get(coin)
        if cached and now - cached[0] < 3600:
            return cached[1]
        data = await self._get_json(CHART.format(coin), {"period1": int(now - 520 * 86400), "period2": int(now) + 86400,
                                                         "interval": "1d", "events": "div|split"})
        res = (data or {}).get("chart", {}).get("result")
        if not res:
            raise RuntimeError(f"Yahoo no devolvió velas para {coin}")
        df = parse_daily(res[0])
        self._candles[coin] = (now, df)
        return df

    async def _spark(self, symbols: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for i in range(0, len(symbols), 20):
            batch = symbols[i:i + 20]
            data = await self._get_json(SPARK, {"symbols": ",".join(batch), "range": "1d", "interval": "5m"})
            for sym, v in (data or {}).items():
                closes, ts = v.get("close") or [], v.get("timestamp") or []
                for c, t in zip(reversed(closes), reversed(ts)):
                    if c is not None:
                        out[sym] = float(c)
                        self.last_quote_ts = max(self.last_quote_ts, float(t))
                        break
        return out

    async def all_mids(self) -> dict[str, float]:
        if time.time() - self._mids_ts < 20 and self._mids:
            return self._mids
        symbols = sorted(self.coins) or ["SPY"]
        fresh = await self._spark(symbols)
        if fresh:
            self._mids = {**self._mids, **fresh}
            self._mids_ts = time.time()
        return self._mids

    async def can_trade(self) -> bool:
        """Mercado abierto (horario regular) y con cotizaciones frescas (descarta festivos)."""
        if not in_regular_hours(ny_now()):
            return False
        await self._spark(["SPY"])
        return time.time() - self.last_quote_ts < 30 * 60

    def market_status(self) -> str:
        now = ny_now()
        if in_regular_hours(now):
            return "abierto"
        return "cerrado (fin de semana)" if now.weekday() >= 5 else "cerrado"


class StockData:
    """Sustituye a DerivativesData: universo (acciones más líquidas) y snapshots."""

    def __init__(self, settings: Settings, feed: StockFeed):
        self.s = settings
        self.feed = feed
        self.snapshots: dict[str, AssetSnapshot] = {}
        self.watchlist: list[str] = []

    def candidates(self) -> list[str]:
        lines = self.s.stock_universe_file.read_text().splitlines()
        return [ln.strip().upper() for ln in lines if ln.strip() and not ln.startswith("#")]

    async def refresh(self) -> dict[str, AssetSnapshot]:
        mids = await self.feed.all_mids() if self.feed.coins else {}
        now = time.time()
        for coin in self.watchlist:
            cached = self.feed._candles.get(coin)
            if not cached or cached[1].empty:
                continue
            df = cached[1]
            px = mids.get(coin, float(df["close"].iloc[-1]))
            self.snapshots[coin] = AssetSnapshot(
                coin=coin, mark_px=px, oracle_px=px, prev_day_px=float(df["close"].iloc[-1]),
                day_ntl_vlm=float((df["close"] * df["volume"]).tail(30).median()), open_interest=0.0,
                funding_hourly=0.0, premium=0.0, max_leverage=self.s.max_leverage, ts=now)
        return self.snapshots

    def oi_change(self, coin: str, window_sec: float = 3600) -> float | None:
        return None

    async def funding_history(self, coin: str, hours: int = 24) -> list[float]:
        return []

    async def build_watchlist(self) -> list[str]:
        async def liquidity(sym: str) -> tuple[str, float]:
            try:
                df = await self.feed.candles(sym)
                return sym, float((df["close"] * df["volume"]).tail(30).median()) if len(df) >= 60 else 0.0
            except Exception as exc:  # noqa: BLE001
                log.warning("Sin datos de %s: %s", sym, exc)
                return sym, 0.0
        ranked = sorted(await asyncio.gather(*(liquidity(s) for s in self.candidates())), key=lambda kv: -kv[1])
        self.watchlist = [s for s, v in ranked if v > 0][: self.s.stock_watchlist_size]
        self.feed.set_ws_coins(self.watchlist)
        log.info("Watchlist acciones (%d): %s", len(self.watchlist), ", ".join(self.watchlist))
        await self.refresh()
        return self.watchlist


class StockNews:
    """Titulares por acción (RSS de Yahoo Finance)."""

    def __init__(self, settings: Settings, session: aiohttp.ClientSession):
        self.s = settings
        self.http = session
        self._cache: dict[str, tuple[float, list[NewsItem]]] = {}

    async def for_coin(self, coin: str, max_items: int = 8, max_age_h: float = 48) -> list[NewsItem]:
        cached = self._cache.get(coin)
        if cached and time.time() - cached[0] < 600:
            return cached[1][:max_items]
        items: list[NewsItem] = []
        try:
            async with self.http.get(RSS, params={"s": coin, "region": "US", "lang": "en-US"}, headers=UA,
                                     timeout=aiohttp.ClientTimeout(total=15)) as r:
                text = await r.text()
            feed = await asyncio.to_thread(feedparser.parse, text)
            cutoff = time.time() - max_age_h * 3600
            for e in feed.entries[:20]:
                pub = None
                if getattr(e, "published_parsed", None):
                    pub = datetime(*e.published_parsed[:6], tzinfo=ZoneInfo("UTC"))
                    if pub.timestamp() < cutoff:
                        continue
                items.append(NewsItem(title=e.get("title", "").strip(), source="Yahoo Finance", published=pub, url=e.get("link", "")))
        except Exception as exc:  # noqa: BLE001
            log.debug("RSS %s falló: %s", coin, exc)
        self._cache[coin] = (time.time(), items)
        return items[:max_items]

    async def latest(self) -> list[NewsItem]:
        return await self.for_coin("SPY", max_items=20)
