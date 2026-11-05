"""Motor de órdenes de Hyperliquid (Paper Trading y Live) y gestión de salidas.

Ciclo de vida de cada posición:
1. Entrada a mercado (IoC agresiva) con el tamaño calculado por RiskManager.
2. Protección inmediata: SL (trigger market, reduce-only, 100 %) y en LIVE un
   TP1 (trigger market, reduce-only, 50 %) residentes en el exchange.
3. TP1 (R:R 1:1.5): se cierra el 50 % y el SL se mueve a breakeven.
4. El 50 % restante sigue un trailing stop a 2x ATR desde el extremo favorable.

En LIVE se opera exclusivamente con una Agent Wallet (clave delegada sin
permisos de retiro) firmando en nombre de `HL_ACCOUNT_ADDRESS`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import Settings
from data.market_feed import MarketFeed
from execution.risk_manager import TradePlan, round_price, round_size
from execution.tax_exporter import LedgerRow, TaxExporter

log = logging.getLogger(__name__)


# ====================================================================== #
# Modelos
# ====================================================================== #
@dataclass
class Fill:
    avg_px: float
    size: float
    fee: float
    oid: int | None = None


@dataclass
class ManagedPosition:
    trade_id: str
    coin: str
    is_long: bool
    size_initial: float
    size: float
    entry_px: float
    stop: float
    initial_stop: float
    tp1: float
    atr: float
    sz_decimals: int
    opened_at: str
    entry_fee: float
    risk_usd: float
    confidence: int = 0
    setup: str = ""
    tp1_hit: bool = False
    extreme_px: float = 0.0
    sl_oid: int | None = None
    tp_oid: int | None = None
    funding_unallocated: float = 0.0
    last_funding_ts: float = field(default_factory=time.time)
    realized_pnl_usd: float = 0.0
    last_fill_check_ms: int = 0

    @property
    def opened_dt(self) -> datetime:
        return datetime.fromisoformat(self.opened_at)

    @property
    def open_risk(self) -> float:
        """Pérdida potencial si salta el stop actual (0 si el stop ya protege la entrada)."""
        loss_per_unit = (self.entry_px - self.stop) if self.is_long else (self.stop - self.entry_px)
        return max(0.0, loss_per_unit * self.size)

    def unrealized(self, mark: float) -> float:
        return (mark - self.entry_px) * self.size * (1 if self.is_long else -1)


@dataclass
class TradeEvent:
    kind: str  # "entry" | "tp1" | "stop_moved" | "exit" | "error"
    coin: str
    data: dict[str, Any] = field(default_factory=dict)


# ====================================================================== #
# Brokers
# ====================================================================== #
class Broker(ABC):
    exchange_side_stops: bool = False

    @abstractmethod
    async def equity(self, marks: dict[str, float]) -> float: ...

    @abstractmethod
    async def market(self, coin: str, is_buy: bool, size: float, reduce_only: bool, ref_px: float) -> Fill | None: ...

    async def place_trigger(self, coin: str, close_is_buy: bool, size: float, trigger_px: float, kind: str, sz_dec: int) -> int | None:
        return None

    async def cancel(self, coin: str, oid: int | None) -> None:
        return None

    async def position_size(self, coin: str) -> float | None:
        return None

    async def fills_since(self, coin: str, since_ms: int) -> list[dict]:
        return []

    async def funding_since(self, coin: str, since_ms: int) -> float:
        return 0.0


class PaperBroker(Broker):
    """Simula ejecuciones con precio de referencia, slippage y comisiones taker."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.path: Path = settings.state_dir / "paper_account.json"
        self.cash = settings.initial_capital
        self.open_upnl_fn = None  # lo inyecta el trader
        if self.path.exists():
            try:
                self.cash = float(json.loads(self.path.read_text())["cash"])
            except Exception:  # noqa: BLE001
                pass

    def _save(self) -> None:
        self.path.write_text(json.dumps({"cash": self.cash}))

    async def equity(self, marks: dict[str, float]) -> float:
        upnl = self.open_upnl_fn(marks) if self.open_upnl_fn else 0.0
        return self.cash + upnl

    async def market(self, coin: str, is_buy: bool, size: float, reduce_only: bool, ref_px: float) -> Fill | None:
        slip = 0.0005  # 5 bps de deslizamiento simulado
        px = ref_px * (1 + slip if is_buy else 1 - slip)
        fee = px * size * self.s.taker_fee_rate
        self.cash -= fee
        self._save()
        return Fill(avg_px=px, size=size, fee=fee)

    def book_pnl(self, pnl: float) -> None:
        self.cash += pnl
        self._save()


class LiveBroker(Broker):
    exchange_side_stops = True

    def __init__(self, settings: Settings, feed: MarketFeed):
        self.s = settings
        self.feed = feed
        self._exchange = None
        self._lev_set: set[str] = set()

    async def exchange(self):
        if self._exchange is None:
            from eth_account import Account
            from hyperliquid.exchange import Exchange

            wallet = Account.from_key(self.s.hl_agent_private_key)
            log.info("Agent wallet %s operando para %s", wallet.address, self.s.hl_account_address)
            self._exchange = await asyncio.to_thread(
                Exchange, wallet, self.s.hl_api_url, None, None, self.s.hl_account_address
            )
        return self._exchange

    async def _ex(self, fn: str, *args: Any, **kwargs: Any) -> Any:
        ex = await self.exchange()
        return await asyncio.to_thread(getattr(ex, fn), *args, **kwargs)

    @staticmethod
    def _status(res: Any) -> dict:
        if not isinstance(res, dict) or res.get("status") != "ok":
            raise RuntimeError(f"Respuesta inválida del exchange: {res}")
        st = res["response"]["data"]["statuses"][0]
        if "error" in st:
            raise RuntimeError(st["error"])
        return st

    async def equity(self, marks: dict[str, float]) -> float:
        state = await self.feed.call("user_state", self.s.hl_account_address)
        return float(state["marginSummary"]["accountValue"])

    async def ensure_leverage(self, coin: str, max_asset_lev: int) -> None:
        if coin in self._lev_set:
            return
        lev = max(1, min(self.s.max_leverage, max_asset_lev))
        try:
            await self._ex("update_leverage", lev, coin, True)
            self._lev_set.add(coin)
        except Exception as exc:  # noqa: BLE001
            log.warning("update_leverage %s falló: %s", coin, exc)

    async def market(self, coin: str, is_buy: bool, size: float, reduce_only: bool, ref_px: float) -> Fill | None:
        start_ms = int(time.time() * 1000) - 2000
        if reduce_only:
            res = await self._ex("market_close", coin, size, None, self.s.slippage)
        else:
            res = await self._ex("market_open", coin, is_buy, size, None, self.s.slippage)
        st = self._status(res)
        filled = st.get("filled")
        if not filled:
            raise RuntimeError(f"Orden IoC no ejecutada: {st}")
        oid = int(filled["oid"])
        fee = 0.0
        for _ in range(3):  # la comisión llega en userFills
            fills = await self.fills_since(coin, start_ms)
            own = [f for f in fills if int(f.get("oid", -1)) == oid]
            if own:
                fee = sum(float(f.get("fee", 0)) for f in own)
                break
            await asyncio.sleep(1)
        return Fill(avg_px=float(filled["avgPx"]), size=float(filled["totalSz"]), fee=fee, oid=oid)

    async def place_trigger(self, coin: str, close_is_buy: bool, size: float, trigger_px: float, kind: str, sz_dec: int) -> int | None:
        # limit_px actúa como tope de deslizamiento de la orden market disparada
        limit = trigger_px * (1.1 if close_is_buy else 0.9)
        order_type = {"trigger": {"triggerPx": round_price(trigger_px, sz_dec), "isMarket": True, "tpsl": kind}}
        res = await self._ex("order", coin, close_is_buy, size, round_price(limit, sz_dec), order_type, True)
        st = self._status(res)
        resting = st.get("resting") or st.get("filled") or {}
        return int(resting["oid"]) if "oid" in resting else None

    async def cancel(self, coin: str, oid: int | None) -> None:
        if oid is None:
            return
        try:
            await self._ex("cancel", coin, oid)
        except Exception as exc:  # noqa: BLE001
            log.debug("cancel %s/%s: %s", coin, oid, exc)

    async def position_size(self, coin: str) -> float | None:
        state = await self.feed.call("user_state", self.s.hl_account_address)
        for p in state.get("assetPositions", []):
            if p["position"]["coin"] == coin:
                return float(p["position"]["szi"])
        return 0.0

    async def fills_since(self, coin: str, since_ms: int) -> list[dict]:
        fills = await self.feed.call("user_fills_by_time", self.s.hl_account_address, since_ms)
        return [f for f in fills or [] if f.get("coin") == coin]

    async def funding_since(self, coin: str, since_ms: int) -> float:
        try:
            rows = await self.feed.call("user_funding_history", self.s.hl_account_address, since_ms)
        except Exception as exc:  # noqa: BLE001
            log.debug("funding history: %s", exc)
            return 0.0
        return sum(float(r["delta"]["usdc"]) for r in rows or [] if r.get("delta", {}).get("coin") == coin)


# ====================================================================== #
# Trader
# ====================================================================== #
class HyperliquidTrader:
    def __init__(self, settings: Settings, feed: MarketFeed, tax: TaxExporter):
        self.s = settings
        self.feed = feed
        self.tax = tax
        self.broker: Broker = LiveBroker(settings, feed) if settings.is_live else PaperBroker(settings)
        if isinstance(self.broker, PaperBroker):
            self.broker.open_upnl_fn = self._upnl
        self.path: Path = settings.state_dir / f"positions_{settings.trading_mode.value}.json"
        self.positions: dict[str, ManagedPosition] = self._load()
        self.closed_today: list[LedgerRow] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Persistencia
    # ------------------------------------------------------------------ #
    def _load(self) -> dict[str, ManagedPosition]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text())
            return {k: ManagedPosition(**v) for k, v in raw.items()}
        except Exception as exc:  # noqa: BLE001
            log.error("No se pudo cargar %s: %s", self.path, exc)
            return {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in self.positions.items()}, indent=2))
        tmp.replace(self.path)

    # ------------------------------------------------------------------ #
    # Consultas
    # ------------------------------------------------------------------ #
    def _upnl(self, marks: dict[str, float]) -> float:
        return sum(p.unrealized(marks.get(p.coin, p.entry_px)) for p in self.positions.values())

    @property
    def open_count(self) -> int:
        return len(self.positions)

    @property
    def open_risk(self) -> float:
        return sum(p.open_risk for p in self.positions.values())

    async def equity(self, marks: dict[str, float]) -> float:
        return await self.broker.equity(marks)

    async def reconcile(self) -> list[str]:
        """Al arrancar en LIVE: detecta posiciones del estado local cerradas en el
        exchange mientras el bot estaba parado, y posiciones no gestionadas."""
        warnings: list[str] = []
        if not self.s.is_live:
            return warnings
        state = await self.feed.call("user_state", self.s.hl_account_address)
        on_exchange = {p["position"]["coin"]: float(p["position"]["szi"]) for p in state.get("assetPositions", [])}
        for coin in list(self.positions):
            if abs(on_exchange.get(coin, 0.0)) == 0:
                warnings.append(f"{coin}: cerrada en el exchange con el bot parado; se registra")
                await self._finalize_live_close(self.positions[coin], "cerrada_offline")
        for coin, szi in on_exchange.items():
            if coin not in self.positions and szi != 0:
                warnings.append(f"{coin}: posición {szi} NO gestionada por HyperAegis")
        return warnings

    # ------------------------------------------------------------------ #
    # Entrada
    # ------------------------------------------------------------------ #
    async def open_position(self, plan: TradePlan, sz_decimals: int, max_asset_lev: int,
                            confidence: int, setup: str) -> TradeEvent:
        async with self._lock:
            if plan.coin in self.positions:
                return TradeEvent("error", plan.coin, {"msg": "ya existe posición en este activo"})
            try:
                if isinstance(self.broker, LiveBroker):
                    await self.broker.ensure_leverage(plan.coin, max_asset_lev)
                fill = await self.broker.market(plan.coin, plan.is_long, plan.size, False, plan.entry_ref)
            except Exception as exc:  # noqa: BLE001
                log.exception("Entrada %s fallida", plan.coin)
                return TradeEvent("error", plan.coin, {"msg": f"entrada fallida: {exc}"})
            if fill is None or fill.size <= 0:
                return TradeEvent("error", plan.coin, {"msg": "sin ejecución"})

            # Recalcular niveles respecto al precio real de ejecución (mismo R)
            r = abs(plan.entry_ref - plan.stop)
            entry = fill.avg_px
            stop = round_price(entry - r if plan.is_long else entry + r, sz_decimals)
            tp1 = round_price(entry + r * self.s.tp1_rr if plan.is_long else entry - r * self.s.tp1_rr, sz_decimals)
            pos = ManagedPosition(
                trade_id=uuid.uuid4().hex[:12], coin=plan.coin, is_long=plan.is_long,
                size_initial=fill.size, size=fill.size, entry_px=entry, stop=stop, initial_stop=stop,
                tp1=tp1, atr=plan.atr, sz_decimals=sz_decimals,
                opened_at=datetime.now(timezone.utc).isoformat(), entry_fee=fill.fee,
                risk_usd=r * fill.size, confidence=confidence, setup=setup, extreme_px=entry,
                last_fill_check_ms=int(time.time() * 1000),
            )
            # Protección en el exchange
            if self.broker.exchange_side_stops:
                try:
                    pos.sl_oid = await self.broker.place_trigger(plan.coin, not plan.is_long, pos.size, stop, "sl", sz_decimals)
                    tp_size = round_size(pos.size * self.s.tp1_close_fraction, sz_decimals)
                    if tp_size > 0:
                        pos.tp_oid = await self.broker.place_trigger(plan.coin, not plan.is_long, tp_size, tp1, "tp", sz_decimals)
                except Exception as exc:  # noqa: BLE001
                    log.critical("No se pudo colocar SL/TP en %s (%s): cerrando por seguridad", plan.coin, exc)
                    self.positions[plan.coin] = pos
                    await self._close_remaining(pos, entry, "sin_proteccion")
                    return TradeEvent("error", plan.coin, {"msg": f"SL no colocado, posición cerrada: {exc}"})
            self.positions[plan.coin] = pos
            self._save()
            return TradeEvent("entry", plan.coin, {"pos": pos, "fill": fill})

    # ------------------------------------------------------------------ #
    # Gestión de salidas
    # ------------------------------------------------------------------ #
    async def manage(self, marks: dict[str, float], atrs: dict[str, float], fundings: dict[str, float]) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        async with self._lock:
            for coin in list(self.positions):
                pos = self.positions[coin]
                mark = marks.get(coin)
                if mark is None:
                    continue
                if coin in atrs and atrs[coin] > 0:
                    pos.atr = atrs[coin]
                try:
                    if self.s.is_live:
                        events += await self._manage_live(pos, mark)
                    else:
                        self._accrue_paper_funding(pos, mark, fundings.get(coin, 0.0))
                        events += await self._manage_paper(pos, mark)
                    if coin in self.positions:
                        events += await self._trail(pos, mark)
                except Exception as exc:  # noqa: BLE001
                    log.exception("Error gestionando %s", coin)
                    events.append(TradeEvent("error", coin, {"msg": str(exc)}))
            self._save()
        return events

    def _accrue_paper_funding(self, pos: ManagedPosition, mark: float, funding_hourly: float) -> None:
        now = time.time()
        hours = (now - pos.last_funding_ts) / 3600
        if hours <= 0:
            return
        # Long paga funding positivo; short lo cobra
        pos.funding_unallocated += -(1 if pos.is_long else -1) * pos.size * mark * funding_hourly * hours
        pos.last_funding_ts = now

    async def _manage_paper(self, pos: ManagedPosition, mark: float) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        hit_stop = mark <= pos.stop if pos.is_long else mark >= pos.stop
        hit_tp1 = mark >= pos.tp1 if pos.is_long else mark <= pos.tp1
        if hit_stop:
            reason = "stop_loss" if not pos.tp1_hit else ("breakeven" if pos.stop == pos.entry_px else "trailing_stop")
            events.append(await self._close_remaining(pos, pos.stop, reason))
        elif hit_tp1 and not pos.tp1_hit:
            events += await self._take_tp1(pos, pos.tp1)
        return events

    async def _manage_live(self, pos: ManagedPosition, mark: float) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        szi = await self.broker.position_size(pos.coin)
        live_size = abs(szi or 0.0)
        if live_size == 0:
            events.append(await self._finalize_live_close(pos, "stop_exchange"))
            return events
        if not pos.tp1_hit and live_size < pos.size - 10 ** (-pos.sz_decimals) / 2:
            # El TP1 residente en el exchange se ha ejecutado
            closed = pos.size - live_size
            fills = await self._closing_fills(pos)
            px = self._vwap(fills) or pos.tp1
            fee = sum(float(f.get("fee", 0)) for f in fills)
            events += await self._after_partial(pos, closed, px, fee, "tp1")
        elif not pos.tp1_hit and pos.tp_oid is None:
            hit_tp1 = mark >= pos.tp1 if pos.is_long else mark <= pos.tp1
            if hit_tp1:
                events += await self._take_tp1(pos, mark)
        return events

    async def _closing_fills(self, pos: ManagedPosition) -> list[dict]:
        fills = await self.broker.fills_since(pos.coin, pos.last_fill_check_ms)
        closing_side = "A" if pos.is_long else "B"
        out = [f for f in fills if f.get("side") == closing_side and "Close" in f.get("dir", "")]
        if fills:
            pos.last_fill_check_ms = max(int(f["time"]) for f in fills) + 1
        return out

    @staticmethod
    def _vwap(fills: list[dict]) -> float | None:
        qty = sum(float(f["sz"]) for f in fills)
        return None if qty == 0 else sum(float(f["px"]) * float(f["sz"]) for f in fills) / qty

    async def _take_tp1(self, pos: ManagedPosition, ref_px: float) -> list[TradeEvent]:
        closed = round_size(pos.size * self.s.tp1_close_fraction, pos.sz_decimals)
        if closed <= 0:
            pos.tp1_hit = True
            return []
        fill = await self.broker.market(pos.coin, not pos.is_long, closed, True, ref_px)
        if fill is None:
            return []
        return await self._after_partial(pos, fill.size, fill.avg_px, fill.fee, "tp1")

    async def _after_partial(self, pos: ManagedPosition, closed: float, px: float, exit_fee: float, reason: str) -> list[TradeEvent]:
        row = await self._record(pos, closed, px, exit_fee, reason)
        pos.size = round_size(pos.size - closed, pos.sz_decimals)
        pos.tp1_hit = True
        old_stop = pos.stop
        pos.stop = round_price(pos.entry_px, pos.sz_decimals)  # breakeven
        await self._replace_stop(pos)
        self._save()
        return [TradeEvent("tp1", pos.coin, {"closed": closed, "px": px, "pnl": row.net_pnl_usd, "new_stop": pos.stop, "old_stop": old_stop})]

    async def _trail(self, pos: ManagedPosition, mark: float) -> list[TradeEvent]:
        if pos.is_long:
            pos.extreme_px = max(pos.extreme_px, mark)
        else:
            pos.extreme_px = min(pos.extreme_px or mark, mark)
        if not pos.tp1_hit or pos.atr <= 0:
            return []
        dist = self.s.trailing_atr_mult * pos.atr
        candidate = pos.extreme_px - dist if pos.is_long else pos.extreme_px + dist
        candidate = round_price(candidate, pos.sz_decimals)
        improves = candidate > pos.stop if pos.is_long else candidate < pos.stop
        # Evita "spam" de órdenes: solo mueve si mejora al menos 0.1 ATR
        if improves and abs(candidate - pos.stop) >= 0.1 * pos.atr:
            old = pos.stop
            pos.stop = candidate
            await self._replace_stop(pos)
            return [TradeEvent("stop_moved", pos.coin, {"old": old, "new": candidate})]
        return []

    async def _replace_stop(self, pos: ManagedPosition) -> None:
        if not self.broker.exchange_side_stops:
            return
        old_oid = pos.sl_oid
        # Primero se coloca el nuevo SL y después se cancela el antiguo: nunca
        # queda la posición sin protección.
        pos.sl_oid = await self.broker.place_trigger(pos.coin, not pos.is_long, pos.size, pos.stop, "sl", pos.sz_decimals)
        await self.broker.cancel(pos.coin, old_oid)

    async def _close_remaining(self, pos: ManagedPosition, ref_px: float, reason: str) -> TradeEvent:
        fill = await self.broker.market(pos.coin, not pos.is_long, pos.size, True, ref_px)
        px = fill.avg_px if fill else ref_px
        fee = fill.fee if fill else 0.0
        if self.broker.exchange_side_stops:
            await self.broker.cancel(pos.coin, pos.sl_oid)
            await self.broker.cancel(pos.coin, pos.tp_oid)
        return await self._finish(pos, pos.size, px, fee, reason)

    async def _finalize_live_close(self, pos: ManagedPosition, reason: str) -> TradeEvent:
        fills = await self._closing_fills(pos)
        px = self._vwap(fills) or pos.stop
        fee = sum(float(f.get("fee", 0)) for f in fills)
        await self.broker.cancel(pos.coin, pos.sl_oid)
        await self.broker.cancel(pos.coin, pos.tp_oid)
        if reason == "stop_exchange":
            reason = "stop_loss" if not pos.tp1_hit else ("breakeven" if pos.stop == pos.entry_px else "trailing_stop")
        return await self._finish(pos, pos.size, px, fee, reason)

    async def _finish(self, pos: ManagedPosition, size: float, px: float, exit_fee: float, reason: str) -> TradeEvent:
        row = await self._record(pos, size, px, exit_fee, reason)
        self.positions.pop(pos.coin, None)
        self._save()
        return TradeEvent("exit", pos.coin, {"reason": reason, "px": px, "pnl": row.net_pnl_usd, "pnl_eur": row.net_pnl_eur,
                                             "total_pnl": pos.realized_pnl_usd})

    async def _record(self, pos: ManagedPosition, size: float, px: float, exit_fee: float, reason: str) -> LedgerRow:
        frac = size / pos.size_initial if pos.size_initial else 1.0
        entry_fee_part = pos.entry_fee * frac
        if self.s.is_live:
            funding = await self.broker.funding_since(pos.coin, int(pos.last_funding_ts * 1000))
            pos.funding_unallocated += funding
            pos.last_funding_ts = time.time()
        funding_part = pos.funding_unallocated * (size / pos.size if pos.size else 1.0)
        pos.funding_unallocated -= funding_part
        row = await self.tax.record_close(
            trade_id=pos.trade_id, coin=pos.coin, is_long=pos.is_long, open_time=pos.opened_dt,
            close_time=datetime.now(timezone.utc), size=size, entry_px=pos.entry_px, exit_px=px,
            fees_usd=entry_fee_part + exit_fee, funding_usd=funding_part, close_reason=reason,
        )
        pos.realized_pnl_usd += row.net_pnl_usd
        if isinstance(self.broker, PaperBroker):
            # En paper las comisiones ya se descontaron en cada ejecución
            gross = (px - pos.entry_px) * size * (1 if pos.is_long else -1)
            self.broker.book_pnl(gross + funding_part)
        self.closed_today.append(row)
        return row

    async def close_all(self, marks: dict[str, float], reason: str) -> list[TradeEvent]:
        events = []
        async with self._lock:
            for coin in list(self.positions):
                pos = self.positions[coin]
                try:
                    events.append(await self._close_remaining(pos, marks.get(coin, pos.entry_px), reason))
                except Exception as exc:  # noqa: BLE001
                    log.exception("close_all %s", coin)
                    events.append(TradeEvent("error", coin, {"msg": str(exc)}))
            self._save()
        return events
