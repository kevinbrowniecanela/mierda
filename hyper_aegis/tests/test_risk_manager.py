from datetime import datetime, timedelta, timezone

from execution.risk_manager import RiskManager, round_price, round_size


def test_rounding():
    assert round_size(1.23456, 2) == 1.23
    assert round_price(65432.123, 5) == 65432.0
    assert round_price(0.123456789, 0) == 0.12346


def test_base_risk_is_one_percent_of_initial(settings):
    rm = RiskManager(settings)
    assert rm.risk_budget(equity=380, open_risk=0, open_count=0) == 4.0


def test_plan_sizing_and_tp1(settings):
    rm = RiskManager(settings)
    plan, why = rm.build_plan(coin="BTC", is_long=True, price=100.0, atr=2.0, proposed_stop=97.0,
                              sz_decimals=3, equity=400, open_risk=0, open_count=0, max_asset_lev=50)
    assert plan, why
    assert plan.stop == 97.0
    assert plan.tp1 == 104.5  # 1.5R
    assert plan.risk_usd <= 4.0 + 1e-9
    assert plan.notional <= 400 * settings.max_leverage


def test_invalid_claude_stop_replaced_by_atr(settings):
    rm = RiskManager(settings)
    plan, _ = rm.build_plan(coin="ETH", is_long=False, price=100.0, atr=2.0, proposed_stop=150.0,
                            sz_decimals=2, equity=400, open_risk=0, open_count=0, max_asset_lev=50)
    assert plan.stop == 103.0  # 1.5 ATR por encima
    assert plan.tp1 == 95.5


def test_house_money_protects_deposit(settings):
    rm = RiskManager(settings)
    rm.update_equity(400)
    assert not rm.state.house_money_active
    rm.update_equity(461)  # +15.25 %
    assert rm.state.house_money_active
    # colchón = 61; con 30 de riesgo abierto y 2 huecos -> min(4.61, 31/2)
    assert abs(rm.risk_budget(461, 30, 1) - 4.61) < 1e-9
    # sin colchón no se opera
    assert rm.risk_budget(410, 10, 1) == 0
    ok, _ = rm.can_open(1, 410, 10)
    assert not ok
    # caer al depósito inicial dispara el shield
    events = rm.update_equity(400)
    assert "shield_breached" in events
    assert not rm.can_open(0, 400, 0)[0]


def test_circuit_breaker_and_max_positions(settings):
    rm = RiskManager(settings)
    now = datetime.now(timezone.utc)
    rm.update_equity(400, now)
    assert rm.can_open(2, 400, 0)[0]
    assert not rm.can_open(3, 400, 0)[0]
    events = rm.update_equity(387, now + timedelta(minutes=5))  # -3.25 %
    assert "circuit_breaker" in events
    assert not rm.can_open(0, 387, 0)[0]


def test_state_persists(settings):
    rm = RiskManager(settings)
    rm.update_equity(470)
    assert RiskManager(settings).state.house_money_active
