"""Gestión de riesgo y "Capital Shield".

Reglas:
- Riesgo por operación: `risk_per_trade_pct` del capital INICIAL (1 % de 400 = 4 USDC).
- House Money: cuando el equity alcanza +15 % sobre el capital inicial, se activa
  (y queda enclavado). Desde ese momento la suma de pérdidas potenciales
  (riesgo abierto) nunca puede llevar el equity por debajo del depósito inicial:
  el riesgo de cada nueva operación sale exclusivamente del colchón de ganancias.
- Tamaño de posición por distancia al stop (basada en ATR / invalidación de Claude).
- TP1 a R:R 1:1.5, máximo 3 posiciones y circuit breaker con drawdown diario > 3 %.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config.settings import Settings

log = logging.getLogger(__name__)


def round_size(size: float, sz_decimals: int) -> float:
    """Redondea el tamaño HACIA ABAJO a los decimales permitidos (nunca aumenta riesgo)."""
    factor = 10**sz_decimals
    return math.floor(size * factor) / factor


def round_price(px: float, sz_decimals: int) -> float:
    """Reglas de precio de perps en Hyperliquid: 5 cifras significativas y
    como máximo (6 - szDecimals) decimales."""
    if px <= 0:
        return px
    return round(float(f"{px:.5g}"), max(0, 6 - sz_decimals))


@dataclass
class TradePlan:
    coin: str
    is_long: bool
    entry_ref: float
    size: float
    stop: float
    tp1: float
    risk_usd: float
    atr: float
    notional: float

    @property
    def r_distance(self) -> float:
        return abs(self.entry_ref - self.stop)


@dataclass
class ShieldState:
    initial_capital: float
    house_money_active: bool = False
    house_money_since: str | None = None
    day: str = ""
    day_start_equity: float = 0.0
    circuit_breaker_day: str | None = None
    shield_breached: bool = False
    peak_equity: float = 0.0
    last_equity: float = 0.0
    history: list[dict] = field(default_factory=list)


class RiskManager:
    def __init__(self, settings: Settings):
        self.s = settings
        self.path: Path = settings.state_dir / "risk_state.json"
        self.state = self._load()

    # ------------------------------------------------------------------ #
    # Persistencia
    # ------------------------------------------------------------------ #
    def _load(self) -> ShieldState:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                st = ShieldState(**data)
                if st.initial_capital != self.s.initial_capital:
                    log.warning(
                        "INITIAL_CAPITAL cambió (%s -> %s); se conserva el del estado. "
                        "Borra state/risk_state.json para reiniciar.", st.initial_capital, self.s.initial_capital,
                    )
                return st
            except Exception as exc:  # noqa: BLE001
                log.error("risk_state.json corrupto (%s); se reinicia", exc)
        return ShieldState(initial_capital=self.s.initial_capital)

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=2))
        tmp.replace(self.path)

    # ------------------------------------------------------------------ #
    # Estado de la cuenta
    # ------------------------------------------------------------------ #
    @property
    def initial(self) -> float:
        return self.state.initial_capital

    @property
    def house_money_threshold(self) -> float:
        return self.initial * (1 + self.s.house_money_trigger_pct)

    def update_equity(self, equity: float, now: datetime | None = None) -> list[str]:
        """Actualiza el estado con el equity actual. Devuelve eventos para notificar."""
        now = now or datetime.now(timezone.utc)
        events: list[str] = []
        st = self.state
        today = now.strftime("%Y-%m-%d")
        if st.day != today:
            if st.day:
                st.history.append({"day": st.day, "start": st.day_start_equity, "end": st.last_equity})
                st.history = st.history[-120:]
                events.append("new_day")
            st.day = today
            st.day_start_equity = equity
        st.last_equity = equity
        st.peak_equity = max(st.peak_equity, equity)

        if not st.house_money_active and equity >= self.house_money_threshold:
            st.house_money_active = True
            st.house_money_since = now.isoformat()
            events.append("house_money_on")
            log.info("HOUSE MONEY activado: equity %.2f >= %.2f", equity, self.house_money_threshold)

        if st.house_money_active and equity <= self.initial and not st.shield_breached:
            st.shield_breached = True
            events.append("shield_breached")
            log.critical("CAPITAL SHIELD: equity %.2f alcanzó el depósito inicial %.2f", equity, self.initial)

        if st.circuit_breaker_day != today and self.daily_drawdown(equity) >= self.s.daily_drawdown_limit_pct:
            st.circuit_breaker_day = today
            events.append("circuit_breaker")
            log.critical("CIRCUIT BREAKER: drawdown diario %.2f%%", self.daily_drawdown(equity) * 100)

        self.save()
        return events

    def daily_drawdown(self, equity: float) -> float:
        start = self.state.day_start_equity
        return 0.0 if start <= 0 else max(0.0, (start - equity) / start)

    @property
    def circuit_breaker_active(self) -> bool:
        return self.state.circuit_breaker_day == datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def reset_shield(self) -> None:
        """Rearme manual tras una violación del Capital Shield."""
        self.state.shield_breached = False
        self.save()

    # ------------------------------------------------------------------ #
    # Presupuesto de riesgo
    # ------------------------------------------------------------------ #
    def risk_budget(self, equity: float, open_risk: float, open_count: int) -> float:
        """Riesgo máximo en USDC para la próxima operación."""
        base = self.initial * self.s.risk_per_trade_pct
        if not self.state.house_money_active:
            return base
        # House Money: el riesgo se calcula sobre las ganancias. El riesgo total
        # abierto nunca puede superar el colchón (equity - depósito inicial).
        cushion = equity - self.initial - open_risk
        if cushion <= 0:
            return 0.0
        slots = max(1, self.s.max_open_positions - open_count)
        on_profits = equity * self.s.risk_per_trade_pct
        return max(0.0, min(on_profits, cushion / slots))

    def can_open(self, open_count: int, equity: float, open_risk: float) -> tuple[bool, str]:
        if self.state.shield_breached:
            return False, "Capital Shield violado: trading detenido hasta rearme manual"
        if self.circuit_breaker_active:
            return False, "Circuit breaker diario activo"
        if open_count >= self.s.max_open_positions:
            return False, f"Máximo de {self.s.max_open_positions} posiciones alcanzado"
        if self.risk_budget(equity, open_risk, open_count) <= 0:
            return False, "Sin colchón de ganancias disponible (House Money)"
        return True, "ok"

    def build_plan(
        self,
        *,
        coin: str,
        is_long: bool,
        price: float,
        atr: float,
        proposed_stop: float | None,
        sz_decimals: int,
        equity: float,
        open_risk: float,
        open_count: int,
        max_asset_lev: int,
    ) -> tuple[TradePlan | None, str]:
        if atr <= 0 or price <= 0:
            return None, "ATR o precio inválidos"

        # 1) Stop: el de Claude si está en el lado correcto y dentro de [0.5, 3] ATR
        stop = None
        if proposed_stop:
            dist = (price - proposed_stop) if is_long else (proposed_stop - price)
            if self.s.min_stop_atr_mult * atr <= dist <= self.s.max_stop_atr_mult * atr:
                stop = proposed_stop
        if stop is None:
            dist = self.s.default_stop_atr_mult * atr
            stop = price - dist if is_long else price + dist
        stop = round_price(stop, sz_decimals)
        r = abs(price - stop)
        if r <= 0:
            return None, "Distancia de stop nula"

        # 2) Tamaño por riesgo (incluye comisiones de entrada+salida)
        risk_usd = self.risk_budget(equity, open_risk, open_count)
        per_unit_loss = r + price * self.s.taker_fee_rate * 2
        size = risk_usd / per_unit_loss

        # 3) Límite de apalancamiento
        lev = min(self.s.max_leverage, max_asset_lev)
        max_notional = equity * lev
        if size * price > max_notional:
            size = max_notional / price
        size = round_size(size, sz_decimals)
        notional = size * price
        if size <= 0 or notional < 10.0:  # mínimo de orden de Hyperliquid: 10 USD
            return None, f"Tamaño insuficiente (nocional {notional:.2f} USD)"

        tp1_dist = r * self.s.tp1_rr
        tp1 = round_price(price + tp1_dist if is_long else price - tp1_dist, sz_decimals)
        plan = TradePlan(
            coin=coin, is_long=is_long, entry_ref=price, size=size, stop=stop, tp1=tp1,
            risk_usd=size * per_unit_loss, atr=atr, notional=notional,
        )
        return plan, "ok"
