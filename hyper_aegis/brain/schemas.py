"""Esquema estricto de la decisión de trading emitida por Claude.

Dos capas de validación:
1. `TRADE_DECISION_JSON_SCHEMA` se envía a la API como structured output
   (`output_config.format`), de modo que Claude solo puede devolver JSON
   con esta forma.
2. `TradeDecision` (Pydantic) re-valida la respuesta y añade reglas de
   negocio (rangos, coherencia del stop con la dirección).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Action(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


class Setup(str, Enum):
    TREND_CONTINUATION = "trend_continuation"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    SQUEEZE = "positioning_squeeze"
    NOISE = "noise"


class TradeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Action
    setup: Setup
    confidence: int = Field(ge=0, le=100, description="Confianza 0-100")
    stop_loss: float | None = Field(default=None, description="Precio de invalidación")
    thesis: str = Field(min_length=1, max_length=1200)
    invalidation: str = Field(min_length=1, max_length=600)
    risk_flags: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _check_stop(self) -> "TradeDecision":
        if self.action != Action.NO_TRADE:
            if self.stop_loss is None or self.stop_loss <= 0:
                raise ValueError("stop_loss es obligatorio y positivo para LONG/SHORT")
        return self

    def stop_is_coherent(self, price: float) -> bool:
        if self.stop_loss is None:
            return False
        if self.action == Action.LONG:
            return self.stop_loss < price
        if self.action == Action.SHORT:
            return self.stop_loss > price
        return True


# Esquema JSON enviado a la API (structured outputs exige
# additionalProperties=false y la lista completa de `required`).
TRADE_DECISION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [a.value for a in Action]},
        "setup": {"type": "string", "enum": [s.value for s in Setup]},
        "confidence": {"type": "integer", "description": "Confianza 0-100"},
        "stop_loss": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "Precio de stop (invalidación). null si NO_TRADE.",
        },
        "thesis": {"type": "string"},
        "invalidation": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "setup", "confidence", "stop_loss", "thesis", "invalidation", "risk_flags"],
    "additionalProperties": False,
}
