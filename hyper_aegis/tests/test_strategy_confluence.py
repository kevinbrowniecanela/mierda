"""Módulo 3: puntuación de confluencia, stop estructural, objetivo y tamaño con riesgo del 1 %."""
import pytest

from strategy.confluence_and_risk import (
    ScoreConfig, build_trade_signal, compute_position_size, evaluate_patterns, floor_size, round_to_tick, score_confluence,
    structural_stop, take_profit_for,
)
from strategy.filters_and_candles import LONG, SHORT, FilterConfig, FunnelResult, PatternSignal, Regime
from tests.strategy_helpers import random_walk, with_ind

FILT, CFG = FilterConfig(), ScoreConfig()
T0 = random_walk(5).index[-1]


def regime(tf, bias, adx=30.0):
    return Regime(tf, adx, "trend" if adx > 25 else "range", bias, 100, 99, 98, 0.01, 0.01)


def funnel(macro="UP", struct="UP", adx=30.0, rel_volume=2.0, funding=1e-5, oi=0.01, penalized=(), allowed=(LONG, SHORT)):
    d = {"oi_change": oi, "price_change": 0.0, "blocked": [], "penalized": list(penalized)}
    if funding is not None:
        d["funding"] = funding
    return FunnelResult("X", True, None, regimes={"4h": regime("4h", macro, 30), "1h": regime("1h", struct, adx)},
                        rel_volume=rel_volume, allowed_directions=set(allowed), derivatives=d)


def pattern(direction=LONG, kind="continuation", entry=100.0, stop_ref=99.0, name="engulfing"):
    return PatternSignal(name, direction, T0, entry, stop_ref, kind, {})


# ---------------------------------------------------------------------- #
# Puntuación
# ---------------------------------------------------------------------- #
def test_perfect_score_is_ten():
    score, reasons, parts = score_confluence(LONG, pattern(), funnel(), FILT, CFG)
    assert score == 10.0 and parts == {"mtf": 2.5, "pattern": 3.0, "volume": 2.0, "derivatives": 1.5, "adx": 1.0}
    assert len(reasons) == 6 and any("MTF" in r for r in reasons)


def test_mtf_partial_flat_and_opposite():
    assert score_confluence(LONG, pattern(), funnel(struct="FLAT"), FILT, CFG)[2]["mtf"] == 1.25  # solo 4h alineada
    assert score_confluence(LONG, pattern(), funnel(macro="DOWN"), FILT, CFG)[2]["mtf"] == 0.0  # un timeframe en contra anula
    assert score_confluence(SHORT, pattern(SHORT), funnel(macro="DOWN", struct="DOWN"), FILT, CFG)[2]["mtf"] == 2.5


def test_volume_and_derivatives_components():
    assert score_confluence(LONG, pattern(), funnel(rel_volume=1.4), FILT, CFG)[2]["volume"] == 0.0
    assert score_confluence(LONG, pattern(), funnel(funding=None, oi=None), FILT, CFG)[2]["derivatives"] == 0.0
    crowded = funnel(penalized=["LONG: funding 0.0200%/h elevado"])
    assert score_confluence(LONG, pattern(), crowded, FILT, CFG)[2]["derivatives"] == 0.75  # solo el OI suma
    assert score_confluence(SHORT, pattern(SHORT), crowded, FILT, CFG)[2]["derivatives"] == 1.5  # la penalización es del LONG


def test_adx_favorable_depends_on_pattern_kind():
    assert score_confluence(LONG, pattern(kind="continuation"), funnel(adx=30), FILT, CFG)[2]["adx"] == 1.0
    assert score_confluence(LONG, pattern(kind="continuation"), funnel(adx=18), FILT, CFG)[2]["adx"] == 0.0
    assert score_confluence(LONG, pattern(kind="reversal"), funnel(adx=18), FILT, CFG)[2]["adx"] == 1.0
    assert score_confluence(LONG, pattern(kind="reversal"), funnel(adx=35), FILT, CFG)[2]["adx"] == 0.0


# ---------------------------------------------------------------------- #
# Stop y objetivo
# ---------------------------------------------------------------------- #
def test_stop_uses_farther_of_wick_and_atr():
    # mecha a 98,0 -> 97,9 con el margen de 0,1 ATR; el stop a 1,5 ATR sería 98,5 -> manda la mecha (más lejos)
    stop, note = structural_stop(LONG, 100.0, pattern(stop_ref=98.0), atr=1.0, cfg=CFG)
    assert stop == pytest.approx(97.9) and "mecha" in note
    # mecha pegada a la entrada: manda 1,5 ATR
    stop, note = structural_stop(LONG, 100.0, pattern(stop_ref=99.9), atr=1.0, cfg=CFG)
    assert stop == pytest.approx(98.5) and "1.5 ATR" in note
    s_stop, _ = structural_stop(SHORT, 100.0, pattern(SHORT, stop_ref=102.0), atr=1.0, cfg=CFG)
    assert s_stop == pytest.approx(102.1)


def test_stop_is_capped_at_three_atr():
    stop, note = structural_stop(LONG, 100.0, pattern(stop_ref=90.0), atr=1.0, cfg=CFG)
    assert stop == pytest.approx(97.0) and "limitado" in note
    s_stop, _ = structural_stop(SHORT, 100.0, pattern(SHORT, stop_ref=110.0), atr=1.0, cfg=CFG)
    assert s_stop == pytest.approx(103.0)


def test_take_profit_ratios():
    tp, rr = take_profit_for(LONG, 100.0, 98.0, 7.0, CFG)
    assert (tp, rr) == (104.0, 2.0)
    tp, rr = take_profit_for(SHORT, 100.0, 102.0, 9.0, CFG)
    assert (tp, rr) == (95.0, 2.5)


# ---------------------------------------------------------------------- #
# Tick y tamaño
# ---------------------------------------------------------------------- #
def test_round_to_tick_and_floor():
    assert round_to_tick(61252.123, 5) == 61252.0  # 5 cifras significativas
    assert round_to_tick(0.0123456, 0) == 0.012346
    assert round_to_tick(4.71374, 1) == 4.7137
    assert floor_size(5.6789, 2) == 5.67 and floor_size(0.0001, 3) == 0.0


def test_position_size_is_one_percent_risk():
    p = compute_position_size(1000.0, entry=100.0, stop=98.0, sz_decimals=2)
    assert p.ok and p.size == 5.0 and p.risk_usd == pytest.approx(10.0) and p.risk_pct == pytest.approx(0.01)
    assert p.notional == 500.0 and p.leverage == pytest.approx(0.5)


def test_size_rounds_down_so_risk_never_exceeds_one_percent():
    p = compute_position_size(224.0, entry=4.7137, stop=4.7555, sz_decimals=1)
    assert p.size == 53.5 and p.risk_usd <= 2.24 + 1e-9 and p.risk_pct <= 0.01
    short = compute_position_size(224.0, entry=4.7137, stop=4.6719, sz_decimals=1)
    assert short.size == 53.5  # la distancia se mide en valor absoluto


def test_leverage_cap_min_notional_fees_and_invalid():
    capped = compute_position_size(100.0, entry=100.0, stop=99.9, sz_decimals=2, max_leverage=5)
    assert capped.ok and capped.notional == pytest.approx(500.0) and "apalancamiento" in capped.reason and capped.risk_pct < 0.01
    tiny = compute_position_size(100.0, entry=100.0, stop=90.0, sz_decimals=2)  # nocional 10 USD justo
    assert tiny.ok and tiny.notional == pytest.approx(10.0)
    small = compute_position_size(50.0, entry=100.0, stop=90.0, sz_decimals=2)
    assert not small.ok and "mínimo" in small.reason
    assert not compute_position_size(100.0, 100.0, 100.0, 2).ok and not compute_position_size(0, 100, 99, 2).ok
    assert not compute_position_size(100.0, entry=100.0, stop=50.0, sz_decimals=0).ok  # tamaño 0 tras redondear
    fee = compute_position_size(1000.0, 100.0, 98.0, 3, fee_rate=0.00045)
    assert fee.size < 5.0 and fee.risk_usd <= 10.0 + 1e-9  # las comisiones reducen el tamaño


# ---------------------------------------------------------------------- #
# Señal completa
# ---------------------------------------------------------------------- #
def test_build_trade_signal_approved_with_position():
    sig = build_trade_signal("UNI", pattern(SHORT, "reversal", entry=4.7137, stop_ref=4.74, name="pin_bar"), funnel(macro="DOWN", struct="DOWN", adx=18),
                             atr=0.0279, sz_decimals=1, account_value=224.0, max_leverage=10, filt=FILT, cfg=CFG)
    assert sig.approved and sig.score == 10.0 and sig.direction == SHORT and sig.symbol == "UNI"
    assert sig.stop_loss > sig.entry_price > sig.take_profit and sig.rr == 2.5  # score >= 8.5
    assert sig.position.ok and sig.position.risk_usd <= 2.24 + 1e-9
    assert abs(sig.take_profit - sig.entry_price) == pytest.approx(2.5 * abs(sig.stop_loss - sig.entry_price), rel=0.02)
    d = sig.to_dict()
    assert d["position"]["size"] == sig.position.size and isinstance(d["reasons"], list)


def test_threshold_boundary_6_5():
    # 3 (patrón) + 2 (volumen) + 1,5 (derivados) = 6,5 -> aprobada; sin ADX y sin MTF
    ok = build_trade_signal("X", pattern(), funnel(macro="FLAT", struct="FLAT", adx=18), 1.0, 2, 1000.0, filt=FILT, cfg=CFG)
    assert ok.score == 6.5 and ok.approved
    low = build_trade_signal("X", pattern(), funnel(macro="FLAT", struct="FLAT", adx=18, rel_volume=1.2), 1.0, 2, 1000.0, filt=FILT, cfg=CFG)
    assert low.score == 4.5 and not low.approved and "puntuación" in low.reject_reason and low.position is None


def test_rejected_when_position_not_viable():
    sig = build_trade_signal("X", pattern(), funnel(), 1.0, 2, account_value=10.0, filt=FILT, cfg=CFG)
    assert sig.score >= 6.5 and not sig.approved and "mínimo" in sig.reject_reason


def test_evaluate_patterns_skips_blocked_directions_and_sorts():
    df = with_ind(random_walk(260, seed=1))
    f = funnel(allowed=(LONG,))
    out = evaluate_patterns("X", [pattern(SHORT), pattern(LONG, kind="reversal", name="pin_bar"), pattern(LONG)], f, df, 2, 1000.0, filt=FILT, cfg=CFG)
    assert [s.direction for s in out] == [LONG, LONG] and out[0].score >= out[1].score
