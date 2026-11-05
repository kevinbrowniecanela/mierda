"""Datos de derivados (Open Interest, Funding) y watchlist dinámica.

Watchlist = Top N por volumen nocional 24h + Top M "movers" del día
(mayor variación absoluta 24h con volumen mínimo), sin duplicados.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from config.settings import Settings
from data.market_feed import MarketFeed

log = logging.getLogger(__name__)


@dataclass
class AssetSnapshot:
    coin: str
    mark_px: float
    oracle_px: float
    prev_day_px: float
    day_ntl_vlm: float
    open_interest: float  # en unidades del activo
    funding_hourly: float  # tasa horaria (Hyperliquid liquida funding cada hora)
    premium: float
    max_leverage: int
    ts: float

    @property
    def change_24h(self) -> float:
        return 0.0 if not self.prev_day_px else self.mark_px / self.prev_day_px - 1

    @property
    def oi_usd(self) -> float:
        return self.open_interest * self.mark_px

    @property
    def funding_annualized(self) -> float:
        return self.funding_hourly * 24 * 365


class DerivativesData:
    def __init__(self, settings: Settings, feed: MarketFeed):
        self.s = settings
        self.feed = feed
        self.snapshots: dict[str, AssetSnapshot] = {}
        # Historial de OI (ts, oi) por moneda para medir cambios de posicionamiento
        self._oi_hist: dict[str, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=240))
        self.watchlist: list[str] = []

    async def refresh(self) -> dict[str, AssetSnapshot]:
        universe, ctxs = await self.feed.meta_and_ctxs()
        now = time.time()
        snaps: dict[str, AssetSnapshot] = {}
        for asset, ctx in zip(universe, ctxs):
            if asset.get("isDelisted") or ctx.get("markPx") is None:
                continue
            coin = asset["name"]
            snap = AssetSnapshot(
                coin=coin,
                mark_px=float(ctx["markPx"]),
                oracle_px=float(ctx.get("oraclePx") or ctx["markPx"]),
                prev_day_px=float(ctx.get("prevDayPx") or 0),
                day_ntl_vlm=float(ctx.get("dayNtlVlm") or 0),
                open_interest=float(ctx.get("openInterest") or 0),
                funding_hourly=float(ctx.get("funding") or 0),
                premium=float(ctx.get("premium") or 0),
                max_leverage=int(asset.get("maxLeverage", 1)),
                ts=now,
            )
            snaps[coin] = snap
            self._oi_hist[coin].append((now, snap.open_interest))
        self.snapshots = snaps
        return snaps

    def oi_change(self, coin: str, window_sec: float = 3600) -> float | None:
        """Variación relativa del OI en la ventana indicada (None si no hay datos)."""
        hist = self._oi_hist.get(coin)
        if not hist or len(hist) < 2:
            return None
        now_ts, now_oi = hist[-1]
        ref = next((oi for ts, oi in hist if now_ts - ts <= window_sec), None)
        if not ref:
            return None
        return now_oi / ref - 1

    async def funding_history(self, coin: str, hours: int = 24) -> list[float]:
        start = int((time.time() - hours * 3600) * 1000)
        rows = await self.feed.call("funding_history", coin, start)
        return [float(r["fundingRate"]) for r in rows or []]

    async def build_watchlist(self) -> list[str]:
        snaps = await self.refresh()
        by_volume = sorted(snaps.values(), key=lambda a: a.day_ntl_vlm, reverse=True)
        top_volume = [a.coin for a in by_volume[: self.s.watchlist_top_volume]]
        movers_pool = [a for a in snaps.values() if a.day_ntl_vlm >= self.s.movers_min_volume_usd and a.coin not in top_volume]
        movers = sorted(movers_pool, key=lambda a: abs(a.change_24h), reverse=True)
        top_movers = [a.coin for a in movers[: self.s.watchlist_top_movers]]
        self.watchlist = top_volume + top_movers
        log.info("Watchlist (%d): %s | movers: %s", len(self.watchlist), ", ".join(top_volume), ", ".join(top_movers))
        return self.watchlist
