"""Modo diario: estrategias, interés compuesto, salida chandelier, vigilante y panel."""
import json

import numpy as np
import pandas as pd
import pytest

from execution.hyperliquid_trader import HyperliquidTrader
from execution.risk_manager import RiskManager
from execution.watchdog import Watchdog
from quant.strategies import STRATEGIES, latest_signal, regime
from tests.test_execution import FixedRateTax


def trending_df(n=400, drift=0.004, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame({"open": close * 0.998, "high": close * 1.01, "low": close * 0.985, "close": close,
                         "volume": rng.uniform(900, 1100, n)}, index=idx)


def test_strategies_are_causal():
    """La señal en la vela i no cambia al añadir velas futuras (sin look-ahead)."""
    df = trending_df()
    for name, fn in STRATEGIES.items():
        full = np.asarray(fn(df)["sig"])
        part = np.asarray(fn(df.iloc[:350])["sig"])
        assert (full[:350] == part).all(), name


def test_latest_signal_and_regime():
    df = trending_df()
    assert regime(df) == "alcista"
    df.iloc[-1, df.columns.get_loc("close")] = df["high"].iloc[-60:-1].max() * 1.05  # ruptura clara
    sig = latest_signal("BTC", df)
    assert sig and sig.direction == 1 and "donchian55_tendencia" in sig.strategies and sig.atr > 0
    assert latest_signal("BTC", df.iloc[:100]) is None  # histórico insuficiente


def daily(settings, **over):
    return settings.model_copy(update={"exit_mode": "chandelier", "trailing_atr_mult": 3.0, "default_stop_atr_mult": 2.0,
                                       "risk_basis": "equity", **over})


def test_risk_compounds_on_equity(settings):
    rm = RiskManager(daily(settings))
    assert rm.risk_budget(equity=400, open_risk=0, open_count=0) == pytest.approx(4.0)
    assert rm.risk_budget(equity=300, open_risk=0, open_count=0) == pytest.approx(3.0)  # se reduce al perder
    rm.halt("prueba")
    assert rm.risk_budget(400, 0, 0) == 0 and not rm.can_open(0, 400, 0)[0]
    rm.reset_shield()
    assert rm.can_open(0, 400, 0)[0]


@pytest.mark.asyncio
async def test_chandelier_exit_trails_from_entry(settings):
    s = daily(settings)
    trader = HyperliquidTrader(s, feed=None, tax=FixedRateTax(s, session=None))
    plan, _ = RiskManager(s).build_plan(coin="SOL", is_long=True, price=100.0, atr=2.0, proposed_stop=None,
                                        sz_decimals=2, equity=400, open_risk=0, open_count=0, max_asset_lev=20)
    assert plan.stop == 96.0  # 2 ATR
    await trader.open_position(plan, 2, 20, 0, "donchian55_tendencia")
    pos = trader.positions["SOL"]
    # Sube a 120: sin cierre parcial; el stop sigue a 3 ATR del máximo
    events = await trader.manage({"SOL": 120.0}, {"SOL": 2.0}, {})
    assert [e.kind for e in events] == ["stop_moved"] and not pos.tp1_hit
    assert pos.stop == pytest.approx(114.0) and pos.size == pos.size_initial
    events = await trader.manage({"SOL": 113.0}, {"SOL": 2.0}, {})
    assert events[-1].kind == "exit" and events[-1].data["reason"] == "trailing_stop"
    assert await trader.equity({}) > 400


class FakeTrader:
    open_risk = 4.0

    def __init__(self):
        self.closed = []
        self.positions = {}

    async def close_all(self, marks, reason, only=None):
        self.closed.append(reason)
        return ["x"]


@pytest.mark.asyncio
async def test_watchdog_pauses_and_kill_switch(settings):
    rm = RiskManager(settings)
    trader = FakeTrader()
    wd = Watchdog(settings, trader, rm, feed=None)
    # 3 errores seguidos de un bucle -> pausa; al recuperarse se levanta
    assert wd.record_error("scan", RuntimeError("x")) is None
    assert wd.record_error("scan", RuntimeError("x")) is None
    inc = wd.record_error("scan", RuntimeError("x"))
    assert inc and inc.kind == "bucle_fallando" and wd.entries_paused
    wd.record_ok("scan")
    assert not wd.entries_paused
    # Caída normal (<5 %): nada. Caída anómala: kill switch
    assert await wd.check(400, {}) == []
    assert await wd.check(392, {}) == []
    incs = await wd.check(350, {})
    assert incs[0].kind == "fuga_equity" and rm.state.halted and trader.closed == ["kill_switch"]
    assert not rm.can_open(0, 350, 0)[0]
    # Datos obsoletos -> pausa
    wd.last_marks_ts -= 10_000
    incs = await wd.check(350, {})
    assert any(i.kind == "datos_obsoletos" for i in incs) and wd.entries_paused
    assert len(wd.recent()) >= 3 and json.dumps(wd.recent())
