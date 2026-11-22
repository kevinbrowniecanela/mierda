"""Modo cartera: aportaciones, bloqueo trimestral de ganancias y Capital Shield."""
from datetime import datetime, timezone

import pytest

from execution.hyperliquid_trader import HyperliquidTrader
from execution.portfolio import PortfolioManager
from execution.risk_manager import RiskManager
from tests.test_execution import FixedRateTax


@pytest.fixture
def pm(settings):
    s = settings.model_copy(update={"market": "stocks", "portfolio_mode": True, "portfolio_core_pct": 0.7,
                                    "portfolio_lock_frac": 0.5, "portfolio_lock_to": "cash", "initial_capital": 100.0})
    risk = RiskManager(s)
    trader = HyperliquidTrader(s, feed=None, tax=FixedRateTax(s, session=None))
    trader.broker.cash = 100.0
    risk.update_equity(100.0, datetime(2026, 11, 20, tzinfo=timezone.utc))
    m = PortfolioManager(s, risk, trader)
    m.init(100.0, datetime(2026, 11, 20, tzinfo=timezone.utc))
    return m


def test_deposit_splits_core_and_sleeve(pm):
    pm.deposit(100.0, core_price=500.0)
    st = pm.state
    assert st.core_units == pytest.approx(70 / 500) and st.core_cost == pytest.approx(70.0)
    assert pm.trader.broker.cash == pytest.approx(130.0)  # 100 + 30 al satélite
    assert pm.risk.initial == pytest.approx(130.0) and st.sleeve_basis == pytest.approx(130.0)
    assert st.contributed_usd == pytest.approx(200.0)
    with pytest.raises(ValueError):
        pm.deposit(-5, 500)


def test_lock_only_on_new_quarter_with_profit(pm):
    same_q = datetime(2026, 12, 15, tzinfo=timezone.utc)
    assert pm.maybe_lock(150.0, 150.0, 500.0, same_q) is None  # mismo trimestre (T4): no hay traspaso
    nxt = datetime(2027, 1, 2, tzinfo=timezone.utc)
    ev = pm.maybe_lock(150.0, 150.0, 500.0, nxt)  # ganancia 50 -> asegura 25
    assert ev and ev["usd"] == pytest.approx(25.0)
    assert pm.state.vault_usd == pytest.approx(25.0) and pm.trader.broker.cash == pytest.approx(75.0)
    assert pm.risk.initial == pytest.approx(125.0) and not pm.risk.state.house_money_active
    assert pm.state.sleeve_basis == pytest.approx(125.0)
    # mismo trimestre otra vez: nada; trimestre siguiente sin ganancia nueva: nada
    assert pm.maybe_lock(125.0, 75.0, 500.0, datetime(2027, 2, 1, tzinfo=timezone.utc)) is None
    assert pm.maybe_lock(125.0, 75.0, 500.0, datetime(2027, 4, 2, tzinfo=timezone.utc)) is None


def test_lock_limited_by_free_cash_and_no_lock_on_loss(pm):
    nxt = datetime(2027, 1, 2, tzinfo=timezone.utc)
    ev = pm.maybe_lock(150.0, 10.0, 500.0, nxt)  # casi todo invertido: solo hay 10 de efectivo libre
    assert ev["usd"] == pytest.approx(10.0)
    ev2 = pm.maybe_lock(90.0, 90.0, 500.0, datetime(2027, 4, 2, tzinfo=timezone.utc))  # pérdida: no se asegura
    assert ev2 is None


def test_lock_to_core_buys_units(pm):
    pm.s.portfolio_lock_to = "core"
    pm.maybe_lock(160.0, 160.0, 400.0, datetime(2027, 1, 2, tzinfo=timezone.utc))  # ganancia 60 -> 30 al núcleo
    assert pm.state.core_units == pytest.approx(30 / 400) and pm.state.vault_usd == 0
    st = pm.status(400.0, 130.0)
    assert st["total"] == pytest.approx(30 + 130.0) and st["lock_to"] == "core"


def test_state_persists(pm, settings):
    pm.deposit(50.0, 500.0)
    again = PortfolioManager(pm.s, pm.risk, pm.trader)
    assert again.state.core_units == pytest.approx(pm.state.core_units) and len(again.state.events) == 2
