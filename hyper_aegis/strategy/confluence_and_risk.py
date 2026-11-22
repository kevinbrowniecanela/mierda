"""Módulo 3 — Puntuación de confluencia y gestión de riesgo.

* `score_confluence`: puntúa de 0 a 10 una señal (patrón + contexto) con pesos configurables:
  alineación MTF 4h/1h (2,5) + patrón validado (3,0) + volumen relativo > 1,5x (2,0) + funding y OI favorables (1,5)
  + ADX favorable (1,0). Solo se aprueban señales con `score >= 6.5`.
* `structural_stop` / `take_profit_for`: stop detrás de la mecha del patrón o a 1,5 x ATR (el más alejado, con
  tope en 3 x ATR) y objetivo con relación riesgo/beneficio >= 1:2 (1:2,5 si la puntuación es excepcional).
* `compute_position_size`: tamaño con riesgo fijo del 1 % del capital: `size = capital x 1% / |entrada - stop|`,
  redondeado SIEMPRE hacia abajo a `szDecimals`, respetando el apalancamiento máximo y el nocional mínimo.
  Precios redondeados al tick de Hyperliquid (5 cifras significativas y como mucho 6 - szDecimals decimales).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from strategy.filters_and_candles import LONG, SHORT, FilterConfig, FunnelResult, PatternSignal
from strategy.logutil import log_event

log = logging.getLogger(__name__)

MAX_RISK_PER_TRADE = 0.01  # 1 % del capital por operación
MIN_ORDER_NOTIONAL_USD = 10.0  # mínimo de una orden en Hyperliquid


# ---------------------------------------------------------------------- #
# Configuración y estructuras
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoreConfig:
    w_mtf: float = 2.5
    w_pattern: float = 3.0
    w_volume: float = 2.0
    w_derivs: float = 1.5
    w_adx: float = 1.0
    min_score: float = 6.5
    volume_mult: float = 1.5
    adx_trend: float = 25.0
    oi_favorable_floor: float = -0.005  # el OI no cae más de un 0,5 % en la ventana
    # stop / objetivo
    stop_atr_mult: float = 1.5
    wick_buffer_atr: float = 0.1  # el stop va un poco más allá de la mecha
    max_stop_atr: float = 3.0  # máximo lógico del stop
    rr_min: float = 2.0
    rr_high: float = 2.5
    rr_high_score: float = 8.5  # con esta puntuación o más se usa rr_high
    risk_per_trade: float = MAX_RISK_PER_TRADE


@dataclass(frozen=True)
class PositionPlan:
    """Tamaño calculado con riesgo fijo; `ok=False` explica por qué no se puede operar."""

    ok: bool
    size: float = 0.0
    notional: float = 0.0
    risk_usd: float = 0.0
    risk_pct: float = 0.0
    leverage: float = 0.0
    reason: str = ""


@dataclass
class TradeSignal:
    """Señal de trading puntuada, con niveles y (opcionalmente) tamaño."""

    symbol: str
    direction: str  # LONG | SHORT
    score: float
    reasons: List[str]
    entry_price: float
    stop_loss: float
    take_profit: float
    pattern: str = ""
    rr: float = 0.0
    atr: float = 0.0
    approved: bool = False
    reject_reason: Optional[str] = None
    breakdown: Dict[str, float] = field(default_factory=dict)
    position: Optional[PositionPlan] = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["position"] = None if self.position is None else dict(self.position.__dict__)
        return d


# ---------------------------------------------------------------------- #
# Tick y redondeo
# ---------------------------------------------------------------------- #
def round_to_tick(px: float, sz_decimals: int) -> float:
    """Precio válido en Hyperliquid: 5 cifras significativas y como mucho (6 - szDecimals) decimales."""
    if px <= 0:
        return px
    return round(float(f"{px:.5g}"), max(0, 6 - sz_decimals))


def floor_size(size: float, sz_decimals: int) -> float:
    """Redondea el tamaño HACIA ABAJO a `szDecimals` (nunca aumenta el riesgo)."""
    f = 10 ** sz_decimals
    return math.floor(size * f + 1e-9) / f


# ---------------------------------------------------------------------- #
# Puntuación de confluencia
# ---------------------------------------------------------------------- #
def score_confluence(
    direction: str, pattern: PatternSignal, funnel: FunnelResult, filt: Optional[FilterConfig] = None,
    cfg: Optional[ScoreConfig] = None,
) -> Tuple[float, List[str], Dict[str, float]]:
    """Devuelve `(puntuación 0-10, razones, desglose)`. No decide la aprobación: eso lo hace `build_trade_signal`."""
    cfg, filt = cfg or ScoreConfig(), filt or FilterConfig()
    reasons: List[str] = []
    parts: Dict[str, float] = {}
    want = "UP" if direction == LONG else "DOWN"
    label = "alcista" if direction == LONG else "bajista"

    # 1) Alineación multi-timeframe: la mitad de los puntos por timeframe alineado; un timeframe en contra anula
    macro, struct = funnel.regimes[filt.macro_tf], funnel.regimes[filt.structure_tf]
    biases = [macro.bias, struct.bias]
    if any(b not in (want, "FLAT") for b in biases):
        mtf = 0.0
        reasons.append(f"MTF en contra ({filt.macro_tf} {macro.bias}, {filt.structure_tf} {struct.bias}): +0")
    else:
        mtf = cfg.w_mtf / 2 * sum(1 for b in biases if b == want)
        if mtf:
            which = "4h y 1h" if mtf == cfg.w_mtf else ("4h" if macro.bias == want else "1h")
            reasons.append(f"Alineación MTF {label} ({which}): +{mtf:g}")
    parts["mtf"] = mtf

    # 2) Patrón de vela validado (cerrada, con su contexto ya comprobado en el detector)
    parts["pattern"] = cfg.w_pattern
    reasons.append(f"Patrón {pattern.name} validado en vela cerrada ({pattern.kind}): +{cfg.w_pattern:g}")

    # 3) Volumen relativo
    rv = funnel.rel_volume or 0.0
    parts["volume"] = cfg.w_volume if rv > cfg.volume_mult else 0.0
    if parts["volume"]:
        reasons.append(f"Volumen relativo {rv:.1f}x > {cfg.volume_mult}x: +{cfg.w_volume:g}")

    # 4) Funding y Open Interest favorables (mitad de los puntos cada uno)
    d = funnel.derivatives
    half = cfg.w_derivs / 2
    funding_known = "funding" in d
    crowded = any(s.startswith(f"{direction}:") and "funding" in s for s in d.get("penalized", []))
    oi = d.get("oi_change")
    pts = 0.0
    if funding_known and not crowded:
        pts += half
        reasons.append(f"Funding {d['funding']:.4%}/h sin saturación: +{half:g}")
    if oi is not None and oi >= cfg.oi_favorable_floor:
        pts += half
        reasons.append(f"Open Interest {oi:+.2%} sin agotamiento: +{half:g}")
    parts["derivatives"] = pts

    # 5) ADX favorable: tendencia fuerte para continuación; mercado no direccional para reversión
    adx = struct.adx
    if pattern.kind == "continuation" and adx > cfg.adx_trend and struct.bias == want:
        parts["adx"] = cfg.w_adx
        reasons.append(f"ADX 1h {adx:.0f} > {cfg.adx_trend:g} con sesgo a favor: +{cfg.w_adx:g}")
    elif pattern.kind == "reversal" and adx <= cfg.adx_trend:
        parts["adx"] = cfg.w_adx
        reasons.append(f"ADX 1h {adx:.0f} <= {cfg.adx_trend:g} (apto para reversión): +{cfg.w_adx:g}")
    else:
        parts["adx"] = 0.0

    return round(sum(parts.values()), 2), reasons, parts


# ---------------------------------------------------------------------- #
# Stop loss, take profit y tamaño
# ---------------------------------------------------------------------- #
def structural_stop(
    direction: str, entry: float, pattern: PatternSignal, atr: float, cfg: Optional[ScoreConfig] = None
) -> Tuple[float, str]:
    """Stop detrás de la mecha del patrón o a 1,5 x ATR de la entrada: el más alejado (mayor protección
    estructural), sin pasar de `max_stop_atr` x ATR. Devuelve `(stop, nota)`."""
    cfg = cfg or ScoreConfig()
    sign = 1 if direction == LONG else -1
    wick = pattern.stop_ref - sign * cfg.wick_buffer_atr * atr
    atr_stop = entry - sign * cfg.stop_atr_mult * atr
    stop = min(wick, atr_stop) if direction == LONG else max(wick, atr_stop)
    note = "stop tras la mecha del patrón" if stop == wick else f"stop a {cfg.stop_atr_mult:g} ATR"
    cap = entry - sign * cfg.max_stop_atr * atr
    if (direction == LONG and stop < cap) or (direction == SHORT and stop > cap):
        stop, note = cap, f"stop limitado a {cfg.max_stop_atr:g} ATR (la mecha estaba más lejos)"
    return stop, note


def take_profit_for(direction: str, entry: float, stop: float, score: float, cfg: Optional[ScoreConfig] = None) -> Tuple[float, float]:
    """Objetivo con R:R mínimo 1:2 (1:2,5 si `score >= rr_high_score`). Devuelve `(take_profit, rr)`."""
    cfg = cfg or ScoreConfig()
    rr = cfg.rr_high if score >= cfg.rr_high_score else cfg.rr_min
    risk = abs(entry - stop)
    return (entry + rr * risk if direction == LONG else entry - rr * risk), rr


def compute_position_size(
    account_value: float, entry: float, stop: float, sz_decimals: int, risk_pct: float = MAX_RISK_PER_TRADE,
    max_leverage: Optional[float] = None, min_notional: float = MIN_ORDER_NOTIONAL_USD, fee_rate: float = 0.0,
) -> PositionPlan:
    """`size = account_value x risk_pct / |entry - stop|`, redondeado hacia abajo a `sz_decimals`.

    `fee_rate` (opcional, p. ej. 0,00045) suma la comisión de entrada y salida al riesgo por unidad.
    Si el apalancamiento necesario supera `max_leverage`, el tamaño se reduce (y el riesgo baja del 1 %).
    """
    dist = abs(entry - stop)
    if account_value <= 0 or entry <= 0 or dist <= 0:
        return PositionPlan(False, reason="capital, entrada o distancia al stop inválidos")
    per_unit = dist + 2 * fee_rate * entry
    raw = account_value * risk_pct / per_unit
    reason = ""
    if max_leverage and raw * entry > account_value * max_leverage:
        raw = account_value * max_leverage / entry
        reason = f"limitado por apalancamiento máximo {max_leverage:g}x"
    size = floor_size(raw, sz_decimals)
    notional = size * entry
    if size <= 0:
        return PositionPlan(False, reason="tamaño 0 tras redondear a szDecimals")
    if notional < min_notional:
        return PositionPlan(False, size, notional, reason=f"nocional {notional:.2f} USD < mínimo {min_notional:g} USD")
    risk_usd = size * per_unit
    return PositionPlan(True, size, notional, risk_usd, risk_usd / account_value, notional / account_value, reason)


def build_trade_signal(
    symbol: str, pattern: PatternSignal, funnel: FunnelResult, atr: float, sz_decimals: int,
    account_value: Optional[float] = None, max_leverage: Optional[float] = None,
    filt: Optional[FilterConfig] = None, cfg: Optional[ScoreConfig] = None, fee_rate: float = 0.0,
) -> TradeSignal:
    """Puntúa un patrón, calcula stop/objetivo/tamaño y decide la aprobación (`score >= min_score`)."""
    cfg = cfg or ScoreConfig()
    direction = pattern.direction
    score, reasons, parts = score_confluence(direction, pattern, funnel, filt, cfg)
    entry = round_to_tick(pattern.entry_ref, sz_decimals)
    stop, note = structural_stop(direction, pattern.entry_ref, pattern, atr, cfg)
    stop = round_to_tick(stop, sz_decimals)
    if (direction == LONG and stop >= entry) or (direction == SHORT and stop <= entry):  # tick degenerado
        sig = TradeSignal(symbol, direction, score, reasons, entry, stop, entry, pattern.name, 0.0, atr, False, "stop inválido tras redondeo", parts)
        return sig
    tp_raw, rr = take_profit_for(direction, entry, stop, score, cfg)
    tp = round_to_tick(tp_raw, sz_decimals)
    reasons.append(f"{note}; objetivo R:R 1:{rr:g}")
    sig = TradeSignal(symbol, direction, score, reasons, entry, stop, tp, pattern.name, rr, atr, False, None, parts)
    if score < cfg.min_score:
        sig.reject_reason = f"puntuación {score:.1f} < {cfg.min_score:g}"
    elif account_value is not None:
        sig.position = compute_position_size(account_value, entry, stop, sz_decimals, cfg.risk_per_trade, max_leverage, fee_rate=fee_rate)
        if sig.position.ok:
            sig.approved = True
        else:
            sig.reject_reason = sig.position.reason
    else:
        sig.approved = True
    log_event(log, logging.INFO if sig.approved else logging.DEBUG, "signal_scored", symbol=symbol, direction=direction,
              pattern=pattern.name, score=score, approved=sig.approved, reject=sig.reject_reason)
    return sig


def evaluate_patterns(
    symbol: str, patterns: List[PatternSignal], funnel: FunnelResult, trigger_df: pd.DataFrame, sz_decimals: int,
    account_value: Optional[float] = None, max_leverage: Optional[float] = None, filt: Optional[FilterConfig] = None,
    cfg: Optional[ScoreConfig] = None, fee_rate: float = 0.0,
) -> List[TradeSignal]:
    """Puntúa todos los patrones cuya dirección sigue permitida por el embudo; mejor puntuación primero."""
    atr = float(trigger_df["atr14"].iloc[-1])
    out = []
    for p in patterns:
        if p.direction not in funnel.allowed_directions:
            continue
        out.append(build_trade_signal(symbol, p, funnel, atr, sz_decimals, account_value, max_leverage, filt, cfg, fee_rate))
    return sorted(out, key=lambda s: s.score, reverse=True)
