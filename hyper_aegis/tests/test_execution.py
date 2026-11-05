import csv
from datetime import datetime, timezone

import pytest

from execution.hyperliquid_trader import HyperliquidTrader
from execution.risk_manager import RiskManager
from execution.tax_exporter import TaxExporter


class FixedRateTax(TaxExporter):
    async def usd_to_eur(self, when):
        return 0.9, when.strftime("%Y-%m-%d"), "TEST"


@pytest.mark.asyncio
async def test_paper_lifecycle_tp1_breakeven_trailing(settings):
    tax = FixedRateTax(settings, session=None)
    trader = HyperliquidTrader(settings, feed=None, tax=tax)
    rm = RiskManager(settings)
    plan, _ = rm.build_plan(coin="SOL", is_long=True, price=100.0, atr=2.0, proposed_stop=97.0,
                            sz_decimals=2, equity=400, open_risk=0, open_count=0, max_asset_lev=20)
    ev = await trader.open_position(plan, 2, 20, 80, "breakout")
    assert ev.kind == "entry"
    pos = trader.positions["SOL"]
    assert trader.open_risk > 0

    # TP1 -> cierra 50 % y SL a breakeven
    events = await trader.manage({"SOL": pos.tp1 + 0.01}, {"SOL": 2.0}, {"SOL": 0.0})
    assert [e.kind for e in events][0] == "tp1"
    assert pos.tp1_hit and events[0].data["new_stop"] == pytest.approx(pos.entry_px, rel=1e-4)
    # en el mismo tick el trailing (extremo - 2 ATR) ya mejora el breakeven
    assert pos.stop >= pos.entry_px
    assert pos.size == pytest.approx(pos.size_initial / 2, abs=0.01)
    assert trader.open_risk == pytest.approx(0, abs=0.01)

    # Trailing: nuevo máximo en 112 -> stop = 112 - 2*2 = 108
    events = await trader.manage({"SOL": 112.0}, {"SOL": 2.0}, {"SOL": 0.0})
    assert any(e.kind == "stop_moved" for e in events)
    assert pos.stop == pytest.approx(108.0)

    # Retroceso: salta trailing stop
    events = await trader.manage({"SOL": 107.5}, {"SOL": 2.0}, {"SOL": 0.0})
    assert events[-1].kind == "exit" and events[-1].data["reason"] == "trailing_stop"
    assert "SOL" not in trader.positions

    rows = list(csv.DictReader(open(settings.tax_ledger_path)))
    assert [r["close_reason"] for r in rows] == ["tp1", "trailing_stop"]
    assert all(float(r["net_pnl_eur"]) == pytest.approx(float(r["net_pnl_usd"]) * 0.9, abs=0.01) for r in rows)
    eq = await trader.equity({})
    assert eq > 400  # operación ganadora


@pytest.mark.asyncio
async def test_paper_stop_loss_loses_about_one_r(settings):
    trader = HyperliquidTrader(settings, feed=None, tax=FixedRateTax(settings, session=None))
    rm = RiskManager(settings)
    plan, _ = rm.build_plan(coin="ETH", is_long=False, price=100.0, atr=2.0, proposed_stop=103.0,
                            sz_decimals=2, equity=400, open_risk=0, open_count=0, max_asset_lev=20)
    await trader.open_position(plan, 2, 20, 75, "mean_reversion")
    events = await trader.manage({"ETH": 103.5}, {}, {})
    assert events[0].kind == "exit" and events[0].data["reason"] == "stop_loss"
    loss = 400 - await trader.equity({})
    assert 3.0 < loss < 4.6  # ~1R (4 USDC) + deslizamiento simulado


@pytest.mark.asyncio
async def test_tax_row_short(settings):
    tax = FixedRateTax(settings, session=None)
    t = datetime(2026, 3, 2, 15, 30, 12, tzinfo=timezone.utc)
    row = await tax.record_close(trade_id="x", coin="BTC", is_long=False, open_time=t, close_time=t,
                                 size=0.01, entry_px=60000, exit_px=59000, fees_usd=0.5, funding_usd=-0.2,
                                 close_reason="tp1")
    assert row.gross_pnl_usd == pytest.approx(10.0)
    assert row.net_pnl_usd == pytest.approx(9.3)
    assert row.net_pnl_eur == pytest.approx(8.37)
    assert row.close_time_utc == "2026-03-02T15:30:12+00:00"


@pytest.mark.asyncio
async def test_backfill_pending_rows(settings):
    class FlakyTax(TaxExporter):
        ok = False

        async def usd_to_eur(self, when):
            return (0.92, when.strftime("%Y-%m-%d"), "BCE/Frankfurter") if self.ok else (0.0, "", "PENDIENTE")

    tax = FlakyTax(settings, session=None)
    t = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)
    await tax.record_close(trade_id="a", coin="BTC", is_long=True, open_time=t, close_time=t, size=1,
                           entry_px=100, exit_px=110, fees_usd=0, funding_usd=0, close_reason="tp1")
    tax.ok = True
    assert await tax.backfill_pending() == 1
    row = list(csv.DictReader(open(settings.tax_ledger_path)))[0]
    assert float(row["net_pnl_eur"]) == pytest.approx(9.2) and row["rate_source"] == "BCE/Frankfurter"
