"""Módulo 1 — Extracción de datos y métricas de perpetuos de Hyperliquid.

`MarketDataClient` envuelve `hyperliquid.info.Info` (solo lectura; nunca firma ni envía órdenes) y ofrece:

* Velas OHLCV multi-timeframe (`4h` tendencia, `1h` estructura, `15m` gatillo). Cada vela lleva la columna
  booleana `closed`: ``True`` si su cierre (`T`, en el reloj del servidor) ya pasó. Las señales deben calcularse
  solo con velas cerradas (`CandleSet.closed`), nunca con la vela en formación, que se repinta.
* Contexto de perpetuos (`meta_and_asset_ctxs`): funding actual y predicho, Open Interest, volumen 24 h, etc.,
  con un histórico de OI en memoria para calcular su variación porcentual entre lecturas.
* Métricas del libro L2 (spread y profundidad) para descartar pares ilíquidos.
* Watchlist dinámica: Top N por volumen más Top M "movers" del día.

La red (`TESTNET_API_URL` o `MAINNET_API_URL`) sale de `Settings.use_testnet` (variable `USE_TESTNET` del `.env`).
Hay API síncrona y asíncrona (`a...`, ejecutada en un hilo), con reintentos y espera exponencial ante límites (429).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from hyperliquid.info import Info

from config.settings import Settings, get_settings
from strategy.logutil import log_event

log = logging.getLogger(__name__)

INTERVAL_MS: Dict[str, int] = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "1d": 86_400_000,
}
TIMEFRAMES: Tuple[str, ...] = ("4h", "1h", "15m")  # macro, estructura, gatillo
DEFAULT_LIMITS: Dict[str, int] = {"4h": 300, "1h": 300, "15m": 300}
CANDLE_COLUMNS = ["open", "high", "low", "close", "volume", "trades", "close_time", "closed"]


class MarketDataError(RuntimeError):
    """Fallo de red o respuesta inválida de la API tras agotar los reintentos."""


# ---------------------------------------------------------------------- #
# Estructuras de datos
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class PerpContext:
    """Estado actual de un perpetuo (`meta_and_asset_ctxs`)."""

    coin: str
    mark_px: float
    oracle_px: float
    mid_px: Optional[float]
    prev_day_px: float
    funding: float  # tasa de funding horaria actual (Hyperliquid liquida cada hora)
    predicted_funding: Optional[float]  # tasa horaria predicha para la próxima liquidación (HlPerp)
    premium: float
    open_interest: float  # en unidades del activo
    day_ntl_vlm: float  # volumen nocional 24 h en USD
    max_leverage: int
    sz_decimals: int
    fetched_at: float

    @property
    def oi_usd(self) -> float:
        return self.open_interest * self.mark_px

    @property
    def change_24h(self) -> float:
        return self.mark_px / self.prev_day_px - 1 if self.prev_day_px else 0.0


@dataclass(frozen=True)
class L2Metrics:
    """Liquidez del libro: spread y profundidad dentro de `depth_pct` del precio medio."""

    coin: str
    best_bid: float
    best_ask: float
    mid: float
    spread_bps: float
    bid_depth_usd: float
    ask_depth_usd: float
    depth_pct: float
    server_time_ms: int


@dataclass(frozen=True)
class LiquidityPolicy:
    """Umbrales para descartar pares ilíquidos."""

    min_day_volume_usd: float = 5_000_000.0
    max_spread_bps: float = 10.0
    min_depth_usd: float = 20_000.0  # por lado, dentro de ±0,5 %

    def check(self, ctx: PerpContext, l2: Optional[L2Metrics] = None) -> Tuple[bool, str]:
        if ctx.day_ntl_vlm < self.min_day_volume_usd:
            return False, f"volumen 24h {ctx.day_ntl_vlm:,.0f} USD < {self.min_day_volume_usd:,.0f}"
        if l2 is not None:
            if l2.spread_bps > self.max_spread_bps:
                return False, f"spread {l2.spread_bps:.1f} pb > {self.max_spread_bps:.1f}"
            if min(l2.bid_depth_usd, l2.ask_depth_usd) < self.min_depth_usd:
                return False, f"profundidad {min(l2.bid_depth_usd, l2.ask_depth_usd):,.0f} USD < {self.min_depth_usd:,.0f}"
        return True, "ok"


@dataclass
class CandleSet:
    """Velas de un activo en varios timeframes. Cada DataFrame lleva la columna `closed`."""

    coin: str
    frames: Dict[str, pd.DataFrame]
    fetched_at: float = field(default_factory=time.time)

    def closed(self, tf: str) -> pd.DataFrame:
        """Solo velas CERRADAS del timeframe (úsalas para indicadores y señales)."""
        df = self.frames[tf]
        return df[df["closed"]]

    def last_closed(self, tf: str) -> Optional[pd.Series]:
        df = self.closed(tf)
        return df.iloc[-1] if len(df) else None

    def forming(self, tf: str) -> Optional[pd.Series]:
        """La vela en formación (si la hay): sirve para el precio actual, NO para señales."""
        df = self.frames[tf]
        open_rows = df[~df["closed"]]
        return open_rows.iloc[-1] if len(open_rows) else None


# ---------------------------------------------------------------------- #
# Cliente
# ---------------------------------------------------------------------- #
class MarketDataClient:
    """Cliente de datos de mercado de Hyperliquid (solo lectura)."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        api_url: Optional[str] = None,
        info: Optional[Any] = None,
        policy: Optional[LiquidityPolicy] = None,
        pace_sec: float = 0.05,
        max_retries: int = 5,
        oi_history_len: int = 720,
    ) -> None:
        self.settings = settings or get_settings()
        self.api_url = api_url or self.settings.hl_api_url
        self._info = info  # inyectable para pruebas
        self.policy = policy or LiquidityPolicy()
        self.pace_sec = pace_sec
        self.max_retries = max_retries
        self.clock_offset_ms = 0  # reloj del servidor - reloj local
        self._oi_hist: Dict[str, Deque[Tuple[float, float, float]]] = defaultdict(lambda: deque(maxlen=oi_history_len))
        self._sz_decimals: Dict[str, int] = {}
        self._last_call = 0.0

    # ------------------------------------------------------------------ #
    # Infraestructura
    # ------------------------------------------------------------------ #
    @property
    def info(self) -> Any:
        if self._info is None:
            log_event(log, logging.INFO, "info_connect", url=self.api_url)
            self._info = Info(self.api_url, True)  # skip_ws=True: solo REST
        return self._info

    def _call(self, fn_name: str, *args: Any, **kwargs: Any) -> Any:
        """Llama a `Info.<fn_name>` con ritmo mínimo y reintentos exponenciales (429 espera más)."""
        fn = getattr(self.info, fn_name)
        for attempt in range(1, self.max_retries + 1):
            wait = self.pace_sec - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_call = time.monotonic()
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - el SDK lanza tipos variados
                rate_limited = getattr(exc, "status_code", None) == 429
                if attempt == self.max_retries:
                    log_event(log, logging.ERROR, "info_call_failed", fn=fn_name, attempts=attempt, error=repr(exc))
                    raise MarketDataError(f"{fn_name} falló tras {attempt} intentos: {exc!r}") from exc
                sleep = (5.0 if rate_limited else 1.0) * 2 ** (attempt - 1)
                log_event(log, logging.WARNING, "info_call_retry", fn=fn_name, attempt=attempt, rate_limited=rate_limited, sleep=sleep)
                time.sleep(sleep)
        raise MarketDataError(fn_name)  # inalcanzable

    def now_ms(self) -> int:
        """Hora actual según el reloj del servidor (reloj local + desfase medido)."""
        return int(time.time() * 1000) + self.clock_offset_ms

    def sync_clock(self, coin: str = "BTC") -> int:
        """Mide el desfase con el servidor usando el campo `time` del libro L2. Devuelve el desfase en ms."""
        book = self._call("l2_snapshot", coin)
        server = int(book["time"])
        self.clock_offset_ms = server - int(time.time() * 1000)
        log_event(log, logging.INFO, "clock_synced", offset_ms=self.clock_offset_ms)
        return self.clock_offset_ms

    # ------------------------------------------------------------------ #
    # Velas
    # ------------------------------------------------------------------ #
    def get_candles(self, coin: str, interval: str, limit: int = 300) -> pd.DataFrame:
        """Velas OHLCV (índice = apertura UTC) con la columna `closed` (True = cierre ya ocurrido)."""
        if interval not in INTERVAL_MS:
            raise ValueError(f"intervalo no soportado: {interval}")
        end = self.now_ms()
        start = end - INTERVAL_MS[interval] * (limit + 1)
        raw = self._call("candles_snapshot", coin, interval, start, end)
        return self.parse_candles(raw, end)

    @staticmethod
    def parse_candles(raw: Sequence[Mapping[str, Any]], now_ms: int) -> pd.DataFrame:
        """Convierte la respuesta de la API; una vela está cerrada si su `T` (cierre) ya pasó."""
        if not raw:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        df = pd.DataFrame(raw)
        out = pd.DataFrame(
            {
                "open": df["o"].astype(float), "high": df["h"].astype(float), "low": df["l"].astype(float),
                "close": df["c"].astype(float), "volume": df["v"].astype(float),
                "trades": df["n"].astype(int) if "n" in df else 0,
                "close_time": pd.to_datetime(df["T"].astype("int64"), unit="ms", utc=True),
                "closed": df["T"].astype("int64") < now_ms,
            }
        )
        out.index = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
        out.index.name = "open_time"
        return out.sort_index()[CANDLE_COLUMNS]

    def get_candle_set(self, coin: str, limits: Optional[Mapping[str, int]] = None) -> CandleSet:
        """Velas 4h, 1h y 15m de un activo."""
        limits = dict(DEFAULT_LIMITS, **(limits or {}))
        frames = {tf: self.get_candles(coin, tf, limits[tf]) for tf in TIMEFRAMES}
        log_event(log, logging.DEBUG, "candles_fetched", coin=coin, **{f"n_{tf}": len(f) for tf, f in frames.items()})
        return CandleSet(coin=coin, frames=frames)

    # ------------------------------------------------------------------ #
    # Contexto de perpetuos
    # ------------------------------------------------------------------ #
    def _predicted_fundings(self) -> Dict[str, float]:
        """Funding horario predicho por activo (venue HlPerp). Vacío si el endpoint no responde."""
        try:
            raw = self.info.post("/info", {"type": "predictedFundings"})
        except Exception as exc:  # noqa: BLE001
            log_event(log, logging.WARNING, "predicted_funding_unavailable", error=repr(exc))
            return {}
        out: Dict[str, float] = {}
        for coin, venues in raw or []:
            for venue, data in venues:
                if venue == "HlPerp" and data:
                    out[coin] = float(data["fundingRate"]) / float(data.get("fundingIntervalHours", 1) or 1)
        return out

    def get_perp_contexts(self, with_predicted: bool = True) -> Dict[str, PerpContext]:
        """Estado actual de todos los perpetuos; además anota el OI en el histórico en memoria."""
        meta, ctxs = self._call("meta_and_asset_ctxs")
        predicted = self._predicted_fundings() if with_predicted else {}
        now = time.time()
        out: Dict[str, PerpContext] = {}
        for asset, ctx in zip(meta["universe"], ctxs):
            if asset.get("isDelisted") or ctx.get("markPx") is None:
                continue
            coin = asset["name"]
            self._sz_decimals[coin] = int(asset["szDecimals"])
            c = PerpContext(
                coin=coin, mark_px=float(ctx["markPx"]), oracle_px=float(ctx.get("oraclePx") or ctx["markPx"]),
                mid_px=float(ctx["midPx"]) if ctx.get("midPx") else None, prev_day_px=float(ctx.get("prevDayPx") or 0),
                funding=float(ctx.get("funding") or 0), predicted_funding=predicted.get(coin),
                premium=float(ctx.get("premium") or 0), open_interest=float(ctx.get("openInterest") or 0),
                day_ntl_vlm=float(ctx.get("dayNtlVlm") or 0), max_leverage=int(asset.get("maxLeverage", 1)),
                sz_decimals=int(asset["szDecimals"]), fetched_at=now,
            )
            out[coin] = c
            self._oi_hist[coin].append((now, c.open_interest, c.mark_px))
        log_event(log, logging.INFO, "perp_contexts", n=len(out), with_predicted=bool(predicted))
        return out

    def oi_change_pct(self, coin: str, window_sec: float = 3600.0) -> Optional[float]:
        """Variación relativa del OI respecto a la lectura más antigua dentro de `window_sec`.

        Devuelve `None` si no hay al menos dos lecturas en memoria (llama a `get_perp_contexts` periódicamente).
        """
        hist = self._oi_hist.get(coin)
        if not hist or len(hist) < 2:
            return None
        now_ts, now_oi, _ = hist[-1]
        ref = next(((ts, oi) for ts, oi, _ in hist if now_ts - ts <= window_sec), None)
        if ref is None or ref[0] == now_ts or ref[1] <= 0:
            return None
        return now_oi / ref[1] - 1

    def price_change_pct(self, coin: str, window_sec: float = 3600.0) -> Optional[float]:
        """Variación del precio mark en la misma ventana que `oi_change_pct` (lecturas en memoria)."""
        hist = self._oi_hist.get(coin)
        if not hist or len(hist) < 2:
            return None
        now_ts, _, now_px = hist[-1]
        ref = next(((ts, px) for ts, _, px in hist if now_ts - ts <= window_sec), None)
        if ref is None or ref[0] == now_ts or ref[1] <= 0:
            return None
        return now_px / ref[1] - 1

    def seed_oi_history(self, coin: str, readings: Sequence[Tuple[float, float, float]]) -> None:
        """Carga lecturas previas `(timestamp, open_interest, mark_px)`, p. ej. desde disco al arrancar."""
        for r in readings:
            self._oi_hist[coin].append(tuple(r))  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Libro L2
    # ------------------------------------------------------------------ #
    def get_l2_metrics(self, coin: str, depth_pct: float = 0.005) -> L2Metrics:
        """Spread en puntos básicos y profundidad en USD a ±`depth_pct` del precio medio."""
        book = self._call("l2_snapshot", coin)
        bids, asks = book["levels"][0], book["levels"][1]
        if not bids or not asks:
            raise MarketDataError(f"libro L2 vacío para {coin}")
        bid, ask = float(bids[0]["px"]), float(asks[0]["px"])
        mid = (bid + ask) / 2
        lo, hi = mid * (1 - depth_pct), mid * (1 + depth_pct)
        bid_depth = sum(float(lv["px"]) * float(lv["sz"]) for lv in bids if float(lv["px"]) >= lo)
        ask_depth = sum(float(lv["px"]) * float(lv["sz"]) for lv in asks if float(lv["px"]) <= hi)
        return L2Metrics(coin, bid, ask, mid, (ask - bid) / mid * 10_000, bid_depth, ask_depth, depth_pct, int(book.get("time", 0)))

    # ------------------------------------------------------------------ #
    # Watchlist y cuenta
    # ------------------------------------------------------------------ #
    def build_watchlist(
        self, contexts: Optional[Mapping[str, PerpContext]] = None, top_volume: int = 20, top_movers: int = 5,
        movers_min_volume_usd: Optional[float] = None,
    ) -> List[str]:
        """Top `top_volume` por volumen 24 h + Top `top_movers` por variación absoluta (con volumen mínimo)."""
        ctxs = contexts or self.get_perp_contexts(with_predicted=False)
        min_vol = movers_min_volume_usd if movers_min_volume_usd is not None else self.policy.min_day_volume_usd
        by_vol = sorted(ctxs.values(), key=lambda c: c.day_ntl_vlm, reverse=True)
        top = [c.coin for c in by_vol[:top_volume]]
        pool = [c for c in ctxs.values() if c.coin not in top and c.day_ntl_vlm >= min_vol]
        movers = [c.coin for c in sorted(pool, key=lambda c: abs(c.change_24h), reverse=True)[:top_movers]]
        log_event(log, logging.INFO, "watchlist_built", top_volume=top, movers=movers)
        return top + movers

    def get_account_value(self, address: str) -> float:
        """`marginSummary.accountValue` de la cuenta (USDC)."""
        state = self._call("user_state", address)
        return float(state["marginSummary"]["accountValue"])

    def sz_decimals(self, coin: str) -> int:
        if coin not in self._sz_decimals:
            self.get_perp_contexts(with_predicted=False)
        return self._sz_decimals[coin]

    # ------------------------------------------------------------------ #
    # API asíncrona (cada llamada se ejecuta en un hilo)
    # ------------------------------------------------------------------ #
    async def aget_candle_set(self, coin: str, limits: Optional[Mapping[str, int]] = None) -> CandleSet:
        return await asyncio.to_thread(self.get_candle_set, coin, limits)

    async def aget_perp_contexts(self, with_predicted: bool = True) -> Dict[str, PerpContext]:
        return await asyncio.to_thread(self.get_perp_contexts, with_predicted)

    async def aget_l2_metrics(self, coin: str, depth_pct: float = 0.005) -> L2Metrics:
        return await asyncio.to_thread(self.get_l2_metrics, coin, depth_pct)

    async def abuild_watchlist(self, top_volume: int = 20, top_movers: int = 5) -> List[str]:
        return await asyncio.to_thread(self.build_watchlist, None, top_volume, top_movers)
