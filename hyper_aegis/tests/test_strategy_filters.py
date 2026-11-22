"""Módulo 2: indicadores, embudo de pre-filtros, régimen y patrones de vela."""
from dataclasses import replace

import numpy as np
import pytest

from strategy import indicators as ind
from strategy.filters_and_candles import (
    LONG, SHORT, FilterConfig, classify_regime, detect_engulfing, detect_liquidity_sweep, detect_patterns, detect_pin_bar,
    detect_squeeze_breakout, evaluate_funnel, prepare_frames, trend_direction,
)
from strategy.market_data import CandleSet, PerpContext
from tests.strategy_helpers import frame, random_walk, replace_last, with_ind

CFG = FilterConfig()


def ctx(funding=0.00001, oi=1000.0, px=100.0):
    return PerpContext("X", px, px, px, px, funding, None, 0.0, oi, 1e8, 20, 3, 0.0)


def frames_for_funnel(last_volume=300.0, vol=0.004):
    """Tres timeframes de paseo aleatorio; el volumen de la última vela de gatillo se fija a `last_volume`."""
    out = {}
    for tf, freq in (("4h", "4h"), ("1h", "1h"), ("15m", "15min")):
        df = random_walk(260, vol=vol, seed={"4h": 3, "1h": 4, "15m": 5}[tf], freq=freq)
        if tf == "15m":
            df = replace_last(df, volume=last_volume)
        out[tf] = with_ind(df)
    return out


# ---------------------------------------------------------------------- #
# Indicadores
# ---------------------------------------------------------------------- #
@pytest.mark.skipif(not ind.HAS_PANDAS_TA, reason="pandas-ta no instalado")
def test_native_matches_pandas_ta():
    df = random_walk(400, seed=9)
    a, b = ind.add_indicators(df, True), ind.add_indicators(df, False)
    for col in ("ema21", "ema50", "ema200", "atr14", "rsi14", "adx14", "bb_upper", "bb_lower", "rel_volume"):
        assert np.nanmax(np.abs((a[col] - b[col]).iloc[250:])) < 1e-3, col


def test_indicators_are_causal():
    df = random_walk(400, seed=2)
    full, part = ind.add_indicators(df, False), ind.add_indicators(df.iloc[:300], False)
    assert np.allclose(full["rsi14"].iloc[:300].dropna(), part["rsi14"].dropna()) and np.allclose(full["adx14"].iloc[:300].dropna(), part["adx14"].dropna())


def test_relative_volume_uses_previous_20_bars():
    df = frame([100.0] * 40, vols=[100.0] * 39 + [300.0])
    out = ind.add_indicators(df, False)
    assert out["rel_volume"].iloc[-1] == pytest.approx(3.0)  # la propia vela no entra en su media


# ---------------------------------------------------------------------- #
# Velas cerradas
# ---------------------------------------------------------------------- #
def test_prepare_frames_excludes_forming_candle():
    df = random_walk(260, seed=1)
    df.iloc[-1, df.columns.get_loc("closed")] = False  # última vela en formación
    cs = CandleSet("X", {"4h": df, "1h": df, "15m": df})
    frames = prepare_frames(cs)
    assert len(frames["15m"]) == 259 and frames["15m"].index[-1] == df.index[-2]


# ---------------------------------------------------------------------- #
# Embudo
# ---------------------------------------------------------------------- #
def test_funnel_passes_with_volume_and_calm_derivatives():
    r = evaluate_funnel("X", frames_for_funnel(), ctx(), oi_change=0.01, price_change=0.001, cfg=CFG)
    assert r.passed and r.failed_stage is None and r.allowed_directions == {LONG, SHORT}
    assert r.rel_volume > 1.5 and set(r.regimes) == {"4h", "1h"}


def test_funnel_insufficient_data():
    f = frames_for_funnel()
    f["4h"] = f["4h"].iloc[:100]
    r = evaluate_funnel("X", f, ctx(), cfg=CFG)
    assert not r.passed and r.failed_stage == "data"


def test_funnel_volatility_bounds():
    dead = evaluate_funnel("X", frames_for_funnel(vol=0.0001), ctx(), cfg=CFG)
    assert dead.failed_stage == "volatility" and "muerto" in dead.checks[-1].detail
    wild = evaluate_funnel("X", frames_for_funnel(vol=0.05), ctx(), cfg=CFG)
    assert wild.failed_stage == "volatility" and "caótica" in wild.checks[-1].detail


def test_funnel_relative_volume_gate():
    r = evaluate_funnel("X", frames_for_funnel(last_volume=100.0), ctx(), cfg=CFG)
    assert r.failed_stage == "volume"
    assert evaluate_funnel("X", frames_for_funnel(last_volume=100.0), ctx(), cfg=replace(CFG, rel_volume_min=0.5)).passed


def test_funding_blocks_longs_not_shorts_and_vice_versa():
    hi = evaluate_funnel("X", frames_for_funnel(), ctx(funding=0.0004), cfg=CFG)
    assert hi.passed and hi.allowed_directions == {SHORT} and "funding" in hi.derivatives["blocked"][0]
    lo = evaluate_funnel("X", frames_for_funnel(), ctx(funding=-0.0004), cfg=CFG)
    assert lo.allowed_directions == {LONG}
    mid = evaluate_funnel("X", frames_for_funnel(), ctx(funding=0.00015), cfg=CFG)  # entre 0,01 % y 0,03 %: solo penaliza
    assert mid.allowed_directions == {LONG, SHORT} and mid.derivatives["penalized"]


def test_price_up_with_falling_oi_blocks_longs():
    r = evaluate_funnel("X", frames_for_funnel(), ctx(), oi_change=-0.02, price_change=0.01, cfg=CFG)
    assert r.allowed_directions == {SHORT} and "short squeeze" in r.derivatives["blocked"][0]
    r2 = evaluate_funnel("X", frames_for_funnel(), ctx(), oi_change=-0.02, price_change=-0.01, cfg=CFG)
    assert r2.allowed_directions == {LONG}
    unknown = evaluate_funnel("X", frames_for_funnel(), ctx(), oi_change=None, price_change=None, cfg=CFG)
    assert unknown.allowed_directions == {LONG, SHORT}  # sin histórico de OI no se bloquea


def test_both_directions_blocked_fails_funnel():
    r = evaluate_funnel("X", frames_for_funnel(), ctx(funding=0.0005), oi_change=-0.05, price_change=-0.02, cfg=CFG)
    assert not r.passed and r.failed_stage == "derivatives" and not r.allowed_directions


# ---------------------------------------------------------------------- #
# Régimen
# ---------------------------------------------------------------------- #
def test_regime_uptrend_downtrend_and_range():
    up = with_ind(frame(100 + np.arange(300) * 0.5, spread=0.1, freq="1h"))
    r = classify_regime(up, "1h", CFG)
    assert r.bias == "UP" and r.label == "trend" and r.adx > 25
    down = with_ind(frame(300 - np.arange(300) * 0.5, spread=0.1, freq="1h"))
    assert classify_regime(down, "1h", CFG).bias == "DOWN"
    rng = np.random.default_rng(0)
    flat = with_ind(frame(100 + rng.normal(0, 0.3, 300), spread=0.2, freq="1h"))
    assert classify_regime(flat, "1h", CFG).label in ("range", "transition")


def test_trend_direction_needs_both_timeframes():
    from strategy.filters_and_candles import Regime
    mk = lambda b: Regime("x", 30, "trend", b, 1, 1, 1, 0.01, 0.01)  # noqa: E731
    assert trend_direction({"4h": mk("UP"), "1h": mk("UP")}, CFG) == LONG
    assert trend_direction({"4h": mk("DOWN"), "1h": mk("DOWN")}, CFG) == SHORT
    assert trend_direction({"4h": mk("UP"), "1h": mk("FLAT")}, CFG) is None


# ---------------------------------------------------------------------- #
# Patrones
# ---------------------------------------------------------------------- #
def ramp_then_sweep(direction):
    """250 velas laterales, una rampa fuerte (RSI extremo) y una vela que barre el swing y cierra dentro."""
    sign = 1 if direction == SHORT else -1
    base = list(100 + np.random.default_rng(0).normal(0, 0.1, 230))
    ramp = [base[-1] + sign * 0.5 * (i + 1) for i in range(30)]
    df = frame(base + ramp, spread=0.1)
    swing = df["high"].iloc[-21:-1].max() if direction == SHORT else df["low"].iloc[-21:-1].min()
    if direction == SHORT:
        o, c, h, l = swing - 0.4, swing - 0.3, swing + 1.0, swing - 0.5
    else:
        o, c, h, l = swing + 0.4, swing + 0.3, swing + 0.5, swing - 1.0
    last = df.copy()
    last.iloc[-1, [last.columns.get_loc(k) for k in ("open", "high", "low", "close")]] = [o, h, l, c]
    return with_ind(last), swing


@pytest.mark.parametrize("direction", [SHORT, LONG])
def test_liquidity_sweep_detected(direction):
    df, swing = ramp_then_sweep(direction)
    p = detect_liquidity_sweep(df, CFG)
    assert p and p.direction == direction and p.name == "liquidity_sweep" and p.kind == "reversal"
    assert p.details["wick_ratio"] >= 0.6 and p.stop_ref == (df["high"].iloc[-1] if direction == SHORT else df["low"].iloc[-1])


def test_liquidity_sweep_rejections():
    df, swing = ramp_then_sweep(SHORT)
    closes_above = replace_last(df, close=swing + 0.5, open=swing + 0.2)  # cierra POR ENCIMA del swing: ruptura, no barrida
    assert detect_liquidity_sweep(closes_above, CFG) is None
    small_wick = replace_last(df, high=swing + 0.05, open=swing - 0.45, close=swing - 0.02, low=swing - 0.5)
    assert detect_liquidity_sweep(small_wick, CFG) is None  # perfora el swing pero la mecha es < 60 % del rango
    assert detect_liquidity_sweep(replace_last(df, rsi14=50.0).assign(rsi14=50.0), CFG) is None  # sin RSI extremo


def squeeze_frame(direction=LONG, volume=300.0, body=2.0):
    rng = np.random.default_rng(5)
    base = list(100 + np.cumsum(rng.normal(0, 0.5, 240)))
    closes, highs, lows = base[:], [c + 0.5 for c in base], [c - 0.5 for c in base]
    px = closes[-1]
    for r in np.linspace(1.0, 0.08, 20):  # el rango (y el ATR) se contrae hasta mínimos
        closes.append(px + rng.normal(0, 0.01)); highs.append(closes[-1] + r / 2); lows.append(closes[-1] - r / 2)
    sgn = 1 if direction == LONG else -1
    closes.append(closes[-1] + sgn * body); highs.append(max(closes[-1], closes[-2]) + 0.05); lows.append(min(closes[-1], closes[-2]) - 0.05)
    vols = [100.0] * (len(closes) - 1) + [volume]
    return with_ind(frame(closes, highs=highs, lows=lows, vols=vols))


@pytest.mark.parametrize("direction", [LONG, SHORT])
def test_squeeze_breakout_detected(direction):
    p = detect_squeeze_breakout(squeeze_frame(direction), CFG)
    assert p and p.direction == direction and p.kind == "continuation"
    assert p.details["body_atr"] >= 2 and p.details["rel_volume"] >= 2 and 4 <= p.details["bars"] <= 6


def test_squeeze_breakout_requires_volume_body_and_compression():
    assert detect_squeeze_breakout(squeeze_frame(LONG, volume=150.0), CFG) is None  # volumen < 2x
    assert detect_squeeze_breakout(squeeze_frame(LONG, body=0.05), CFG) is None  # cuerpo < 2 ATR
    noisy = with_ind(random_walk(300, vol=0.01, seed=3))
    assert detect_squeeze_breakout(replace_last(noisy, volume=1000.0), CFG) is None  # sin compresión previa


def engulfing_frame(direction=LONG):
    sgn = 1 if direction == LONG else -1
    closes = list(100 + sgn * np.arange(250) * 0.3)  # tendencia sostenida
    for _ in range(3):  # retroceso hacia la EMA(21)
        closes.append(closes[-1] - sgn * 1.2)
    df = frame(closes, spread=0.1)
    prv = df.iloc[-1]
    cur_open, cur_close = prv["close"] - sgn * 0.1, prv["open"] + sgn * 0.3  # envuelve el cuerpo previo
    ext = max(cur_open, cur_close) + 0.1 if direction == LONG else min(cur_open, cur_close) - 0.1
    ext2 = min(cur_open, cur_close) - 0.1 if direction == LONG else max(cur_open, cur_close) + 0.1
    new = frame(closes + [cur_close], spread=0.1)
    new.iloc[-1, [new.columns.get_loc(k) for k in ("open", "high", "low", "close")]] = [cur_open, ext if direction == LONG else ext2, ext2 if direction == LONG else ext, cur_close]
    return with_ind(new)


@pytest.mark.parametrize("direction", [LONG, SHORT])
def test_engulfing_with_trend_and_pullback(direction):
    df = engulfing_frame(direction)
    p = detect_engulfing(df, direction, CFG)
    assert p and p.direction == direction and p.name == "engulfing"
    assert detect_engulfing(df, None, CFG) is None  # sin tendencia 4h/1h alineada no hay envolvente válida
    assert detect_engulfing(df, SHORT if direction == LONG else LONG, CFG) is None  # contra la tendencia


def pin_frame(direction):
    rng = np.random.default_rng(4)
    closes = list(100 + rng.normal(0, 0.15, 260))
    df = frame(closes, spread=0.1)
    c = 100.0
    vals = [c, c + 0.1, c - 1.6, c + 0.04] if direction == LONG else [c, c + 1.6, c - 0.1, c - 0.04]  # open, high, low, close
    df.iloc[-1, [df.columns.get_loc(k) for k in ("open", "high", "low", "close")]] = vals
    return with_ind(df)


@pytest.mark.parametrize("direction", [LONG, SHORT])
def test_pin_bar_in_key_zone(direction):
    p = detect_pin_bar(pin_frame(direction), CFG)
    assert p and p.direction == direction and p.details["body_ratio"] <= 0.25 and p.details["wick_body"] >= 2.5


def test_pin_bar_rejections():
    df = pin_frame(LONG)
    fat_body = replace_last(df, open=100.9, close=100.0)  # cuerpo grande: no es pin bar
    assert detect_pin_bar(fat_body, CFG) is None
    short_wick = replace_last(df, low=99.95)  # mecha de rechazo < 2,5 x el cuerpo
    assert detect_pin_bar(short_wick, CFG) is None


def test_no_repainting_signals_depend_only_on_closed_candles():
    import pandas as pd
    df = pin_frame(LONG)
    sig_before = detect_patterns(df, None, CFG)
    assert sig_before and sig_before[0].name == "pin_bar"
    # llegan dos velas posteriores: la señal de la vela anterior, calculada con los datos hasta ella, no cambia
    raw = df[["open", "high", "low", "close", "volume", "trades", "closed", "close_time"]]
    nxt = raw.iloc[[-1, -1]].copy()
    nxt.index = raw.index[-1] + pd.to_timedelta([15, 30], unit="min")
    for col, v in (("open", 100.0), ("close", 100.2), ("high", 100.3), ("low", 99.9)):
        nxt[col] = v
    extended = with_ind(pd.concat([raw, nxt]))
    assert detect_patterns(extended.iloc[: len(df)], None, CFG) == sig_before
    assert detect_patterns(extended, None, CFG) == []  # la última vela nueva no es un patrón


def test_detect_patterns_returns_all_present():
    out = detect_patterns(pin_frame(SHORT), None, CFG)
    assert [p.name for p in out] == ["pin_bar"] and out[0].direction == SHORT
