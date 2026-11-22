"""Modo acciones: horario, velas completas, solo largos y entrada con el mercado abierto."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data.stock_feed import in_regular_hours, parse_daily
from quant.strategies import Signal

NY = ZoneInfo("America/New_York")


def test_regular_hours():
    assert not in_regular_hours(datetime(2026, 11, 21, 11, 0, tzinfo=NY))  # sábado
    assert not in_regular_hours(datetime(2026, 11, 23, 9, 29, tzinfo=NY))
    assert in_regular_hours(datetime(2026, 11, 23, 9, 30, tzinfo=NY))
    assert in_regular_hours(datetime(2026, 11, 23, 15, 59, tzinfo=NY))
    assert not in_regular_hours(datetime(2026, 11, 23, 16, 0, tzinfo=NY))


def result(days):
    ts = [int(pd.Timestamp(d, tz="UTC").timestamp()) + 13 * 3600 + 1800 for d in days]
    n = len(ts)
    q = {"open": [100.0] * n, "high": [102.0] * n, "low": [99.0] * n, "close": [101.0] * n, "volume": [1000] * n}
    return {"timestamp": ts, "indicators": {"quote": [q], "adjclose": [{"adjclose": [50.5] * n}]}}


def test_parse_daily_adjusts_and_drops_open_session():
    r = result(["2026-11-18", "2026-11-19", "2026-11-20"])
    during = datetime(2026, 11, 20, 11, 0, tzinfo=NY)
    df = parse_daily(r, now=during)
    assert len(df) == 2 and df.index[-1].date().isoformat() == "2026-11-19"  # sesión de hoy descartada
    assert df["close"].iloc[0] == pytest.approx(50.5) and df["open"].iloc[0] == pytest.approx(50.0)  # ajustada x0.5
    after = datetime(2026, 11, 20, 17, 0, tzinfo=NY)
    assert len(parse_daily(r, now=after)) == 3  # cerrada: se conserva
    weekend = datetime(2026, 11, 22, 12, 0, tzinfo=NY)
    assert len(parse_daily(r, now=weekend)) == 3


@pytest.fixture
def app(settings):
    import run
    s = settings.model_copy(update={"market": "stocks", "allow_short": False, "min_order_usd": 1.0, "max_leverage": 1,
                                    "exit_mode": "chandelier", "anthropic_api_key": "", "dashboard_enabled": False,
                                    "trailing_atr_mult": 3.0, "default_stop_atr_mult": 2.0, "risk_basis": "equity"})
    a = run.HyperAegis(s, session=None)
    a.feed.coins = {"AAA"}
    return a


def signal(direction):
    return Signal(coin="AAA", direction=direction, strategies=["donchian55_tendencia"], bar_time=pd.Timestamp("2026-11-20", tz="UTC"),
                  close=100.0, atr=2.5, stop_k=2.0, trail_k=3.0)


@pytest.mark.asyncio
async def test_short_signal_discarded(app):
    app.feed.can_trade = lambda: _async(True)
    await app.handle_signal(signal(-1))
    assert not app.trader.positions and app._seen["AAA"]  # marcada como vista
    assert "cortos desactivados" in app._decisions_path.read_text()


@pytest.mark.asyncio
async def test_closed_market_defers_without_marking_seen(app):
    app.feed.can_trade = lambda: _async(False)
    await app.handle_signal(signal(1))
    assert not app.trader.positions and "AAA" not in app._seen  # se reintentará con el mercado abierto


@pytest.mark.asyncio
async def test_open_market_enters_long_without_leverage(app):
    app.feed.can_trade = lambda: _async(True)
    app.feed.all_mids = lambda: _async({"AAA": 100.2})
    await app.handle_signal(signal(1))
    pos = app.trader.positions["AAA"]
    assert pos.is_long and pos.stop == pytest.approx(pos.entry_px - 5.0, abs=0.06)  # stop a 2 ATR
    assert pos.size * pos.entry_px <= app.equity * 1.0 + 1e-6  # sin apalancamiento
    assert pos.risk_usd == pytest.approx(app.equity * 0.01, rel=0.05)


async def _async(v):
    return v
