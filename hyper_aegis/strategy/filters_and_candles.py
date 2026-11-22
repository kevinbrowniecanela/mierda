"""Módulo 2 — Filtros cuantitativos y patrones de velas.

Todo se calcula SOLO sobre velas cerradas (`CandleSet.closed`), de modo que una señal no cambia al añadirse
datos posteriores (sin repintado). Dos bloques:

1. **Embudo de pre-filtros** (`evaluate_funnel`): descarta por orden de coste
   datos insuficientes → volatilidad operable (ATR/precio) → volumen relativo → régimen (ADX, EMA 50/200 en 4h y 1h)
   → derivados (funding extremo y Open Interest). El de derivados no descarta el activo, sino la DIRECCIÓN.
2. **Patrones de vela** sobre el timeframe de gatillo (`detect_patterns`): barrida de liquidez, ruptura tras
   compresión, envolvente con tendencia y pin bar en zona clave.

Convenciones: `LONG`/`SHORT`; el funding es la tasa HORARIA de Hyperliquid (0,0003 = 0,03 %/h, ya extremo).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from strategy.indicators import add_indicators
from strategy.logutil import log_event
from strategy.market_data import CandleSet, PerpContext

log = logging.getLogger(__name__)

LONG, SHORT = "LONG", "SHORT"
MIN_BARS = 210  # EMA(200) + margen


# ---------------------------------------------------------------------- #
# Configuración
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class FilterConfig:
    trigger_tf: str = "15m"
    structure_tf: str = "1h"
    macro_tf: str = "4h"
    # volumen y volatilidad
    rel_volume_min: float = 1.5  # volumen > 1,5 x SMA(volumen, 20)
    atr_pct_bounds: Mapping[str, Tuple[float, float]] = field(
        default_factory=lambda: {"15m": (0.0012, 0.025), "1h": (0.003, 0.05)}
    )  # ATR(14)/cierre aceptable por timeframe: por debajo, las comisiones se comen el rango; por encima, caos
    # régimen
    adx_trend: float = 25.0
    adx_range: float = 20.0
    slope_lookback: int = 10  # velas para medir la pendiente de las EMA
    flat_slope_pct: float = 0.0005  # |pendiente| menor que 0,05 % = plana
    # derivados
    funding_block: float = 0.0003  # 0,03 %/h: bloquea largos (y, simétrico, cortos con -0,03 %/h)
    funding_penalty: float = 0.0001  # 0,01 %/h: ya penaliza (posicionamiento saturado)
    oi_window_sec: float = 3600.0
    price_move_pct: float = 0.003  # el precio se mueve > 0,3 % en la ventana...
    oi_drop_pct: float = -0.01  # ...mientras el OI cae más de un 1 %
    # patrones
    sweep_lookback: int = 20
    sweep_wick_ratio: float = 0.6
    sweep_rsi_high: float = 70.0
    sweep_rsi_low: float = 30.0
    squeeze_min_bars: int = 4
    squeeze_max_bars: int = 6
    squeeze_range_atr: float = 0.8  # cada vela de la compresión tiene un rango <= 0,8 x ATR
    squeeze_atr_lookback: int = 20
    squeeze_atr_tol: float = 1.05  # ATR "en mínimos de 20 periodos" con un 5 % de tolerancia
    squeeze_body_atr: float = 2.0
    squeeze_vol_mult: float = 2.0
    engulf_pullback_bars: int = 3
    engulf_ema_tol_atr: float = 0.75  # el retroceso llega a <= 0,75 ATR de la EMA(21)
    pin_body_ratio: float = 0.25
    pin_wick_body_mult: float = 2.5
    pin_touch_tol_atr: float = 0.15


# ---------------------------------------------------------------------- #
# Estructuras
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Regime:
    """Clasificación de tendencia/rango de un timeframe."""

    tf: str
    adx: float
    label: str  # "trend" (ADX > 25) | "range" (ADX < 20) | "transition"
    bias: str  # "UP" | "DOWN" | "FLAT"
    close: float
    ema50: float
    ema200: float
    ema50_slope: float
    ema200_slope: float


@dataclass
class FunnelResult:
    symbol: str
    passed: bool
    failed_stage: Optional[str]
    checks: List[Check] = field(default_factory=list)
    regimes: Dict[str, Regime] = field(default_factory=dict)
    rel_volume: Optional[float] = None
    allowed_directions: Set[str] = field(default_factory=lambda: {LONG, SHORT})
    derivatives: Dict[str, Any] = field(default_factory=dict)  # funding, OI, bloqueos y penalizaciones


@dataclass(frozen=True)
class PatternSignal:
    """Patrón detectado en la última vela cerrada del timeframe de gatillo."""

    name: str  # liquidity_sweep | squeeze_breakout | engulfing | pin_bar
    direction: str
    bar_time: pd.Timestamp
    entry_ref: float  # cierre de la vela del patrón
    stop_ref: float  # extremo estructural (mecha) del patrón
    kind: str  # "reversal" | "continuation"
    details: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Preparación e indicadores
# ---------------------------------------------------------------------- #
def prepare_frames(candles: CandleSet, timeframes: Sequence[str] = ("4h", "1h", "15m")) -> Dict[str, pd.DataFrame]:
    """Indicadores sobre velas CERRADAS de cada timeframe (la vela en formación nunca entra)."""
    return {tf: add_indicators(candles.closed(tf)) for tf in timeframes}


def _slope(series: pd.Series, lookback: int) -> float:
    if len(series) <= lookback or not series.iloc[-1 - lookback]:
        return 0.0
    return float(series.iloc[-1] / series.iloc[-1 - lookback] - 1)


def classify_regime(df: pd.DataFrame, tf: str, cfg: FilterConfig) -> Regime:
    """Régimen: ADX (tendencia/rango) y sesgo por precio, EMA(50), EMA(200) y sus pendientes."""
    last = df.iloc[-1]
    s50, s200 = _slope(df["ema50"], cfg.slope_lookback), _slope(df["ema200"], cfg.slope_lookback)
    adx = float(last["adx14"])
    label = "trend" if adx > cfg.adx_trend else "range" if adx < cfg.adx_range else "transition"
    up = last["close"] > last["ema50"] > last["ema200"] and s50 > -cfg.flat_slope_pct and s200 >= -cfg.flat_slope_pct
    down = last["close"] < last["ema50"] < last["ema200"] and s50 < cfg.flat_slope_pct and s200 <= cfg.flat_slope_pct
    bias = "UP" if up else "DOWN" if down else "FLAT"
    return Regime(tf, adx, label, bias, float(last["close"]), float(last["ema50"]), float(last["ema200"]), s50, s200)


def trend_direction(regimes: Mapping[str, Regime], cfg: FilterConfig) -> Optional[str]:
    """Dirección con alineación multi-timeframe (4h y 1h con el mismo sesgo), o `None`."""
    macro, struct = regimes[cfg.macro_tf].bias, regimes[cfg.structure_tf].bias
    if macro == "UP" and struct == "UP":
        return LONG
    if macro == "DOWN" and struct == "DOWN":
        return SHORT
    return None


# ---------------------------------------------------------------------- #
# 1. Embudo de pre-filtros
# ---------------------------------------------------------------------- #
def evaluate_funnel(
    symbol: str,
    frames: Mapping[str, pd.DataFrame],
    ctx: Optional[PerpContext] = None,
    oi_change: Optional[float] = None,
    price_change: Optional[float] = None,
    cfg: Optional[FilterConfig] = None,
) -> FunnelResult:
    """Aplica el embudo en orden de coste y se detiene en la primera etapa que falla.

    `frames` viene de `prepare_frames`. `oi_change`/`price_change` son variaciones relativas en la misma ventana
    (`MarketDataClient.oi_change_pct` / `price_change_pct`). El resultado indica qué direcciones siguen permitidas.
    """
    cfg = cfg or FilterConfig()
    res = FunnelResult(symbol=symbol, passed=False, failed_stage=None)

    def fail(stage: str, name: str, detail: str) -> FunnelResult:
        res.checks.append(Check(name, False, detail))
        res.failed_stage = stage
        log_event(log, logging.DEBUG, "funnel_discard", symbol=symbol, stage=stage, detail=detail)
        return res

    # etapa 0: datos suficientes
    for tf in (cfg.macro_tf, cfg.structure_tf, cfg.trigger_tf):
        n = len(frames.get(tf, ()))
        if n < MIN_BARS:
            return fail("data", f"datos_{tf}", f"{n} velas cerradas < {MIN_BARS}")
    res.checks.append(Check("datos", True, "velas cerradas suficientes en 4h/1h/15m"))

    # etapa 1: volatilidad operable (ATR / cierre) en gatillo y estructura
    for tf in (cfg.trigger_tf, cfg.structure_tf):
        last = frames[tf].iloc[-1]
        pct = float(last["atr14"] / last["close"])
        lo, hi = cfg.atr_pct_bounds[tf]
        if not lo <= pct <= hi:
            kind = "mercado muerto" if pct < lo else "volatilidad caótica"
            return fail("volatility", f"atr_pct_{tf}", f"ATR/precio {pct:.3%} fuera de [{lo:.2%}, {hi:.2%}] ({kind})")
    res.checks.append(Check("volatilidad", True, "ATR/precio dentro del rango operable"))

    # etapa 2: volumen relativo en la vela de gatillo
    trig = frames[cfg.trigger_tf].iloc[-1]
    res.rel_volume = float(trig["rel_volume"]) if pd.notna(trig["rel_volume"]) else 0.0
    if res.rel_volume <= cfg.rel_volume_min:
        return fail("volume", "volumen_relativo", f"{res.rel_volume:.2f}x <= {cfg.rel_volume_min}x SMA(20)")
    res.checks.append(Check("volumen_relativo", True, f"{res.rel_volume:.2f}x SMA(20)"))

    # etapa 3: régimen de mercado (informa, no descarta; el rango se trata en la puntuación)
    for tf in (cfg.macro_tf, cfg.structure_tf):
        res.regimes[tf] = classify_regime(frames[tf], tf, cfg)
    res.checks.append(Check("regimen", True, ", ".join(f"{tf}: ADX {r.adx:.0f} {r.label}/{r.bias}" for tf, r in res.regimes.items())))

    # etapa 4: derivados -> direcciones permitidas
    _apply_derivatives(res, ctx, oi_change, price_change, cfg)
    if not res.allowed_directions:
        res.failed_stage = "derivatives"
        res.checks.append(Check("derivados", False, "ambas direcciones bloqueadas: " + "; ".join(res.derivatives.get("blocked", []))))
        return res
    res.checks.append(Check("derivados", True, f"direcciones permitidas: {sorted(res.allowed_directions)}"))
    res.passed = True
    return res


def _apply_derivatives(
    res: FunnelResult, ctx: Optional[PerpContext], oi_change: Optional[float], price_change: Optional[float], cfg: FilterConfig
) -> None:
    blocked: List[str] = []
    penalized: List[str] = []
    info: Dict[str, Any] = {"oi_change": oi_change, "price_change": price_change}
    if ctx is not None:
        f = ctx.funding
        info["funding"], info["funding_predicted"] = f, ctx.predicted_funding
        if f > cfg.funding_block:
            res.allowed_directions.discard(LONG)
            blocked.append(f"LONG: funding {f:.4%}/h > {cfg.funding_block:.4%}/h")
        elif f > cfg.funding_penalty:
            penalized.append(f"LONG: funding {f:.4%}/h elevado")
        if f < -cfg.funding_block:
            res.allowed_directions.discard(SHORT)
            blocked.append(f"SHORT: funding {f:.4%}/h < -{cfg.funding_block:.4%}/h")
        elif f < -cfg.funding_penalty:
            penalized.append(f"SHORT: funding {f:.4%}/h muy negativo")
    if oi_change is not None and price_change is not None:
        if price_change >= cfg.price_move_pct and oi_change <= cfg.oi_drop_pct:
            res.allowed_directions.discard(LONG)
            blocked.append(f"LONG: precio {price_change:+.2%} con OI {oi_change:+.2%} (short squeeze agotándose)")
        if price_change <= -cfg.price_move_pct and oi_change <= cfg.oi_drop_pct:
            res.allowed_directions.discard(SHORT)
            blocked.append(f"SHORT: precio {price_change:+.2%} con OI {oi_change:+.2%} (capitulación de largos agotándose)")
    info["blocked"], info["penalized"] = blocked, penalized
    res.derivatives = info


# ---------------------------------------------------------------------- #
# 2. Patrones de vela (sobre la última vela CERRADA del timeframe de gatillo)
# ---------------------------------------------------------------------- #
def _parts(bar: pd.Series) -> Tuple[float, float, float, float]:
    """(rango, cuerpo, mecha superior, mecha inferior)."""
    body_hi, body_lo = max(bar["open"], bar["close"]), min(bar["open"], bar["close"])
    return float(bar["high"] - bar["low"]), float(body_hi - body_lo), float(bar["high"] - body_hi), float(body_lo - bar["low"])


def detect_liquidity_sweep(df: pd.DataFrame, cfg: FilterConfig) -> Optional[PatternSignal]:
    """Barrida de liquidez (fakeout).

    El máximo/mínimo supera el swing de las últimas 20 velas, la vela CIERRA dentro del rango previo y deja una
    mecha de rechazo >= 60 % del rango, con RSI(14) > 70 (corto) o < 30 (largo). Como el RSI suele alcanzar su
    extremo en la vela previa al rechazo, se toma el máximo/mínimo de las últimas 3 velas cerradas.
    """
    n = cfg.sweep_lookback
    if len(df) < n + 3:
        return None
    last, prev = df.iloc[-1], df.iloc[-1 - n : -1]
    rng, _, upper, lower = _parts(last)
    if rng <= 0:
        return None
    swing_hi, swing_lo = float(prev["high"].max()), float(prev["low"].min())
    rsi_recent = df["rsi14"].iloc[-3:]
    if last["high"] > swing_hi and swing_lo <= last["close"] < swing_hi and upper / rng >= cfg.sweep_wick_ratio and rsi_recent.max() > cfg.sweep_rsi_high:
        return PatternSignal("liquidity_sweep", SHORT, df.index[-1], float(last["close"]), float(last["high"]), "reversal",
                             {"swing": swing_hi, "wick_ratio": upper / rng, "rsi": float(rsi_recent.max())})
    if last["low"] < swing_lo and swing_lo < last["close"] <= swing_hi and lower / rng >= cfg.sweep_wick_ratio and rsi_recent.min() < cfg.sweep_rsi_low:
        return PatternSignal("liquidity_sweep", LONG, df.index[-1], float(last["close"]), float(last["low"]), "reversal",
                             {"swing": swing_lo, "wick_ratio": lower / rng, "rsi": float(rsi_recent.min())})
    return None


def detect_squeeze_breakout(df: pd.DataFrame, cfg: FilterConfig) -> Optional[PatternSignal]:
    """Compresión + expansión.

    4 a 6 velas previas de rango estrecho (<= 0,8 x ATR) con el ATR(14) en mínimos de 20 periodos, seguidas de una
    vela de ruptura con cuerpo >= 2 x ATR(14) (el de la vela anterior, antes de que la propia ruptura lo infle),
    volumen >= 2 x SMA(volumen, 20) y cierre fuera del rango comprimido.
    """
    if len(df) < cfg.squeeze_atr_lookback + cfg.squeeze_max_bars + 3:
        return None
    last = df.iloc[-1]
    atr_prev = float(df["atr14"].iloc[-2])
    atr_min = float(df["atr14"].iloc[-1 - cfg.squeeze_atr_lookback : -1].min())
    if atr_prev <= 0 or atr_prev > atr_min * cfg.squeeze_atr_tol:
        return None
    body = abs(float(last["close"] - last["open"]))
    if body < cfg.squeeze_body_atr * atr_prev or not (last["rel_volume"] >= cfg.squeeze_vol_mult):
        return None
    for nbars in range(cfg.squeeze_max_bars, cfg.squeeze_min_bars - 1, -1):
        window = df.iloc[-1 - nbars : -1]
        if (window["high"] - window["low"]).max() > cfg.squeeze_range_atr * atr_prev:
            continue
        hi, lo = float(window["high"].max()), float(window["low"].min())
        meta = {"bars": float(nbars), "body_atr": body / atr_prev, "rel_volume": float(last["rel_volume"])}
        if last["close"] > last["open"] and last["close"] > hi:
            return PatternSignal("squeeze_breakout", LONG, df.index[-1], float(last["close"]), float(last["low"]), "continuation", meta)
        if last["close"] < last["open"] and last["close"] < lo:
            return PatternSignal("squeeze_breakout", SHORT, df.index[-1], float(last["close"]), float(last["high"]), "continuation", meta)
        return None
    return None


def detect_engulfing(df: pd.DataFrame, trend: Optional[str], cfg: FilterConfig) -> Optional[PatternSignal]:
    """Envolvente institucional a favor de la tendencia 4h/1h, tras un retroceso hasta la EMA(21).

    El cuerpo actual envuelve por completo el anterior (color contrario) en la dirección de `trend`; antes hubo un
    retroceso (`engulf_pullback_bars` cierres en contra) y el mínimo/máximo reciente llegó a <= 0,75 ATR de la EMA(21).
    """
    if trend is None or len(df) < cfg.engulf_pullback_bars + 4:
        return None
    cur, prv = df.iloc[-1], df.iloc[-2]
    atr = float(cur["atr14"])
    k = cfg.engulf_pullback_bars
    recent = df.iloc[-1 - k :]  # velas del retroceso + la actual
    if trend == LONG:
        engulfs = prv["close"] < prv["open"] and cur["close"] > cur["open"] and cur["open"] <= prv["close"] and cur["close"] >= prv["open"]
        pulled_back = df["close"].iloc[-2] < df["close"].iloc[-2 - k]
        near_ema = float(recent["low"].min()) <= float(cur["ema21"]) + cfg.engulf_ema_tol_atr * atr
        if engulfs and pulled_back and near_ema and float(cur["close"]) > float(cur["ema21"]):
            return PatternSignal("engulfing", LONG, df.index[-1], float(cur["close"]), float(min(cur["low"], prv["low"])), "continuation",
                                 {"body_ratio": abs(cur["close"] - cur["open"]) / max(abs(prv["close"] - prv["open"]), 1e-12)})
    else:
        engulfs = prv["close"] > prv["open"] and cur["close"] < cur["open"] and cur["open"] >= prv["close"] and cur["close"] <= prv["open"]
        pulled_back = df["close"].iloc[-2] > df["close"].iloc[-2 - k]
        near_ema = float(recent["high"].max()) >= float(cur["ema21"]) - cfg.engulf_ema_tol_atr * atr
        if engulfs and pulled_back and near_ema and float(cur["close"]) < float(cur["ema21"]):
            return PatternSignal("engulfing", SHORT, df.index[-1], float(cur["close"]), float(max(cur["high"], prv["high"])), "continuation",
                                 {"body_ratio": abs(cur["close"] - cur["open"]) / max(abs(prv["close"] - prv["open"]), 1e-12)})
    return None


def detect_pin_bar(df: pd.DataFrame, cfg: FilterConfig) -> Optional[PatternSignal]:
    """Pin bar en zona clave: cuerpo <= 25 % del rango, mecha de rechazo >= 2,5 x el cuerpo y contacto con la banda
    de Bollinger (inferior para largos, superior para cortos) o con la EMA(50)."""
    last = df.iloc[-1]
    rng, body, upper, lower = _parts(last)
    if rng <= 0 or body > cfg.pin_body_ratio * rng:
        return None
    tol = cfg.pin_touch_tol_atr * float(last["atr14"])
    floor = max(body, 1e-12)
    if lower >= cfg.pin_wick_body_mult * floor and lower > upper:
        touch = last["low"] <= last["bb_lower"] + tol or abs(last["low"] - last["ema50"]) <= tol or last["low"] <= last["ema50"] <= last["close"]
        if touch:
            return PatternSignal("pin_bar", LONG, df.index[-1], float(last["close"]), float(last["low"]), "reversal",
                                 {"wick_body": lower / floor, "body_ratio": body / rng})
    if upper >= cfg.pin_wick_body_mult * floor and upper > lower:
        touch = last["high"] >= last["bb_upper"] - tol or abs(last["high"] - last["ema50"]) <= tol or last["close"] <= last["ema50"] <= last["high"]
        if touch:
            return PatternSignal("pin_bar", SHORT, df.index[-1], float(last["close"]), float(last["high"]), "reversal",
                                 {"wick_body": upper / floor, "body_ratio": body / rng})
    return None


def detect_patterns(
    trigger_df: pd.DataFrame, trend: Optional[str] = None, cfg: Optional[FilterConfig] = None
) -> List[PatternSignal]:
    """Todos los patrones presentes en la última vela cerrada del timeframe de gatillo (con indicadores)."""
    cfg = cfg or FilterConfig()
    found = [
        detect_liquidity_sweep(trigger_df, cfg),
        detect_squeeze_breakout(trigger_df, cfg),
        detect_engulfing(trigger_df, trend, cfg),
        detect_pin_bar(trigger_df, cfg),
    ]
    out = [p for p in found if p is not None]
    for p in out:
        log_event(log, logging.INFO, "pattern_detected", pattern=p.name, direction=p.direction, bar=str(p.bar_time), **p.details)
    return out
