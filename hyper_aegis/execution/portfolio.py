"""Modo cartera (acciones, simulación): núcleo indexado + satélite del bot + caja de ganancias.

  Aportación  -> `core_pct` al núcleo (SPY) y el resto al satélite (capital del bot).
  Cada trimestre, si el satélite vale más que lo aportado a él, se traspasa `lock_frac` de la
  ganancia al núcleo (`lock_to=core`) o a una caja en efectivo (`lock_to=cash`, no se invierte).
  Tras cada traspaso el Capital Shield del satélite se rebasa al capital que queda.

Estudio (research/portfolio_formula.py, 2010-2026): ninguna fórmula domina; bloquear a efectivo
cuesta ~0,7 puntos de rentabilidad anual y no reduce la caída del conjunto. Son mandos de riesgo,
no una fuente de rentabilidad.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config.settings import Settings

log = logging.getLogger(__name__)


def quarter(now: datetime) -> str:
    return f"{now.year}-Q{(now.month - 1) // 3 + 1}"


@dataclass
class PortfolioState:
    core_units: float = 0.0
    core_cost: float = 0.0
    vault_usd: float = 0.0
    sleeve_basis: float = 0.0
    contributed_usd: float = 0.0
    last_quarter: str = ""
    events: list[dict] = field(default_factory=list)


class PortfolioManager:
    def __init__(self, settings: Settings, risk, trader):
        self.s = settings
        self.risk = risk
        self.trader = trader
        self.path: Path = settings.state_dir / "portfolio.json"
        self.state = self._load()

    def _load(self) -> PortfolioState:
        if self.path.exists():
            try:
                return PortfolioState(**json.loads(self.path.read_text()))
            except Exception as exc:  # noqa: BLE001
                log.error("portfolio.json corrupto (%s); se reinicia", exc)
        return PortfolioState()

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=2))
        tmp.replace(self.path)

    def _event(self, kind: str, usd: float, detail: str, now: datetime | None = None) -> dict:
        ev = {"ts": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), "kind": kind,
              "usd": round(usd, 2), "detail": detail}
        self.state.events.append(ev)
        self.state.events = self.state.events[-200:]
        return ev

    def init(self, sleeve_equity: float, now: datetime | None = None) -> None:
        """Primer arranque: el capital actual del bot es el satélite inicial."""
        st = self.state
        if st.events:
            return
        now = now or datetime.now(timezone.utc)
        st.sleeve_basis = sleeve_equity
        st.contributed_usd = sleeve_equity
        st.last_quarter = quarter(now)
        self._event("init", sleeve_equity, f"Satélite inicial {sleeve_equity:.2f} USD", now)
        self.save()

    def deposit(self, usd: float, core_price: float, now: datetime | None = None) -> dict:
        """Aportación: reparte entre núcleo (compra de `core_symbol`) y satélite (efectivo del bot)."""
        if usd <= 0 or core_price <= 0:
            raise ValueError("aportación y precio deben ser positivos")
        st = self.state
        core_part = usd * self.s.portfolio_core_pct
        sleeve_part = usd - core_part
        units = core_part / core_price
        st.core_units += units
        st.core_cost += core_part
        st.contributed_usd += usd
        st.sleeve_basis += sleeve_part
        self.trader.broker.add_cash(sleeve_part)
        self.risk.add_capital(sleeve_part)
        ev = self._event("deposit", usd, f"{units:.4f} {self.s.portfolio_core_symbol} por {core_part:.2f} USD; {sleeve_part:.2f} USD al satélite", now)
        self.save()
        return ev

    def maybe_lock(self, sleeve_equity: float, free_cash: float, core_price: float, now: datetime | None = None) -> dict | None:
        """Al cambiar de trimestre, traspasa parte de la ganancia del satélite. Devuelve el evento o None."""
        now = now or datetime.now(timezone.utc)
        st = self.state
        q = quarter(now)
        if not st.last_quarter:
            st.last_quarter = q
            self.save()
            return None
        if q == st.last_quarter:
            return None
        st.last_quarter = q
        profit = sleeve_equity - st.sleeve_basis
        ev = None
        if profit > 0 and self.s.portfolio_lock_frac > 0:
            amount = min(self.s.portfolio_lock_frac * profit, free_cash)
            if amount >= 1.0:
                self.trader.broker.add_cash(-amount)
                self.risk.rebase(sleeve_equity - amount, withdrawn=amount)
                st.sleeve_basis = sleeve_equity - amount
                if self.s.portfolio_lock_to == "core" and core_price > 0:
                    st.core_units += amount / core_price
                    st.core_cost += amount
                    where = f"al núcleo ({self.s.portfolio_core_symbol})"
                else:
                    st.vault_usd += amount
                    where = "a la caja (efectivo)"
                ev = self._event("lock", amount, f"Ganancia del satélite {profit:.2f} USD: {amount:.2f} USD {where}", now)
        self.save()
        return ev

    def status(self, core_price: float | None, sleeve_equity: float) -> dict:
        st = self.state
        core_value = st.core_units * core_price if core_price else st.core_cost
        total = core_value + st.vault_usd + sleeve_equity
        return {
            "core_symbol": self.s.portfolio_core_symbol, "core_units": st.core_units, "core_value": core_value,
            "core_cost": st.core_cost, "vault": st.vault_usd, "sleeve": sleeve_equity, "total": total,
            "contributed": st.contributed_usd, "gain": total - st.contributed_usd,
            "gain_pct": (total / st.contributed_usd - 1) * 100 if st.contributed_usd else 0.0,
            "core_pct": self.s.portfolio_core_pct, "lock_frac": self.s.portfolio_lock_frac,
            "lock_to": self.s.portfolio_lock_to, "events": list(reversed(st.events[-12:])),
        }
