"""Fase 1: filtro cuantitativo que decide cuándo despertar a Claude.

Cada activo recibe una puntuación sumando señales independientes. Solo si
la puntuación supera el umbral (y el activo no está en cooldown, y no se ha
agotado el presupuesto horario de llamadas a Claude) se escala a la Fase 2.
Este módulo no tiene coste de IA.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from config.settings import Settings
from data.derivatives_data import AssetSnapshot
from data.market_feed import L2Book
from quant.technicals import TechSnapshot


@dataclass
class AnomalyReport:
    coin: str
    score: float
    triggered: bool
    reasons: list[str] = field(default_factory=list)
    bias: str = "neutral"  # sesgo direccional sugerido por las señales

    def summary(self) -> str:
        return f"{self.coin} score={self.score:.1f} sesgo={self.bias} :: " + "; ".join(self.reasons)


class AnomalyTrigger:
    def __init__(self, settings: Settings):
        self.s = settings
        self._last_trigger: dict[str, float] = {}
        self._claude_calls: deque[float] = deque()

    def evaluate(
        self,
        coin: str,
        tech: TechSnapshot,
        deriv: AssetSnapshot | None,
        oi_change_1h: float | None,
        book: L2Book | None,
        now: float | None = None,
    ) -> AnomalyReport:
        s = self.s
        score = 0.0
        reasons: list[str] = []
        bull = bear = 0.0

        # 1) Spike de volumen
        if tech.vol_ratio >= s.vol_spike_mult:
            pts = min(2.0, tech.vol_ratio / s.vol_spike_mult)
            score += pts
            reasons.append(f"volumen {tech.vol_ratio:.1f}x media20")
            if tech.ret_1 > 0:
                bull += pts
            else:
                bear += pts

        # 2) Expansión de rango vs ATR
        if tech.range_atr >= s.range_atr_mult:
            score += 1.5
            reasons.append(f"rango vela {tech.range_atr:.1f}x ATR")
            if tech.ret_1 > 0:
                bull += 1
            else:
                bear += 1

        # 3) Cambio de Open Interest
        if oi_change_1h is not None and abs(oi_change_1h) >= s.oi_change_pct:
            score += 1.5
            reasons.append(f"OI {oi_change_1h:+.1%} en 1h")
            # OI sube + precio sube = longs nuevos; OI sube + precio cae = shorts nuevos
            if oi_change_1h > 0:
                if tech.ret_4 > 0:
                    bull += 1
                else:
                    bear += 1

        # 4) Funding extremo (posicionamiento saturado => riesgo de squeeze)
        if deriv and abs(deriv.funding_hourly) >= s.funding_extreme_hourly:
            score += 1.0
            reasons.append(f"funding {deriv.funding_hourly:+.4%}/h ({deriv.funding_annualized:+.0%} anual)")
            if deriv.funding_hourly > 0:
                bear += 0.5  # longs pagando caro: sesgo contrarian
            else:
                bull += 0.5

        # 5) Desequilibrio del libro L2
        if book:
            imb = book.imbalance()
            if abs(imb) >= s.book_imbalance_threshold:
                score += 1.0
                reasons.append(f"libro L2 desequilibrio {imb:+.2f}")
                if imb > 0:
                    bull += 1
                else:
                    bear += 1

        # 6) RSI extremo
        if tech.rsi >= 75 or tech.rsi <= 25:
            score += 0.75
            reasons.append(f"RSI extremo {tech.rsi:.0f}")

        # 7) Cruce de VWAP con impulso
        prev_close = tech.close / (1 + tech.ret_1) if tech.ret_1 > -1 else tech.close
        if (tech.close > tech.vwap) != (prev_close > tech.vwap):
            score += 0.5
            reasons.append("cruce de VWAP")

        # Confluencia con la tendencia de fondo
        if tech.trend == "alcista":
            bull += 0.5
        elif tech.trend == "bajista":
            bear += 0.5

        bias = "long" if bull > bear + 0.5 else "short" if bear > bull + 0.5 else "neutral"
        triggered = score >= s.anomaly_score_threshold and not self._in_cooldown(coin, now)
        return AnomalyReport(coin=coin, score=score, triggered=triggered, reasons=reasons, bias=bias)

    # ------------------------------------------------------------------ #
    def _in_cooldown(self, coin: str, now: float | None = None) -> bool:
        last = self._last_trigger.get(coin, 0)
        return (now or time.time()) - last < self.s.anomaly_cooldown_min * 60

    def claude_budget_available(self, now: float | None = None) -> bool:
        now = now or time.time()
        while self._claude_calls and now - self._claude_calls[0] > 3600:
            self._claude_calls.popleft()
        return len(self._claude_calls) < self.s.max_claude_calls_per_hour

    def mark_escalated(self, coin: str, now: float | None = None) -> None:
        now = now or time.time()
        self._last_trigger[coin] = now
        self._claude_calls.append(now)
