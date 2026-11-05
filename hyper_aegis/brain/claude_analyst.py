"""Fase 2: analista Claude con salida estructurada y validación Pydantic.

- Cliente asíncrono oficial `anthropic.AsyncAnthropic`.
- Salida restringida con `output_config.format` (JSON Schema) y re-validada
  con Pydantic (`TradeDecision`).
- System prompt con `cache_control` para abaratar llamadas repetidas.
- Fallback server-side ante `stop_reason == "refusal"` (beta), y cualquier
  respuesta no válida se convierte en NO_TRADE: el bot nunca opera sobre
  una salida que no haya pasado la validación.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import anthropic
from pydantic import ValidationError

from brain.schemas import TRADE_DECISION_JSON_SCHEMA, Action, Setup, TradeDecision
from config.prompts import SYSTEM_PROMPT, build_user_prompt
from config.settings import Settings

log = logging.getLogger(__name__)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class AnalystResult:
    decision: TradeDecision
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    error: str | None = None


def _no_trade(reason: str) -> TradeDecision:
    return TradeDecision(
        action=Action.NO_TRADE, setup=Setup.NOISE, confidence=0, stop_loss=None,
        thesis=reason[:1200] or "sin decisión", invalidation="n/a", risk_flags=["analyst_error"],
    )


class ClaudeAnalyst:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key or None, max_retries=3, timeout=120.0
        )

    async def analyze(self, context: dict, news_lines: list[str]) -> AnalystResult:
        user_prompt = build_user_prompt(
            json.dumps(context, ensure_ascii=False, indent=1, default=str), "\n".join(news_lines)
        )
        params: dict = dict(
            model=self.s.claude_model,
            max_tokens=self.s.claude_max_tokens,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
            output_config={
                "effort": self.s.claude_effort,
                "format": {"type": "json_schema", "schema": TRADE_DECISION_JSON_SCHEMA},
            },
        )
        try:
            if self.s.claude_enable_fallbacks:
                response = await self.client.beta.messages.create(
                    **params, betas=[_FALLBACK_BETA], fallbacks="default"
                )
            else:
                response = await self.client.messages.create(**params)
        except anthropic.RateLimitError as exc:
            log.warning("Claude rate limit: %s", exc)
            return AnalystResult(_no_trade("rate limit de la API de Claude"), self.s.claude_model, error="rate_limit")
        except anthropic.APIStatusError as exc:
            log.error("Claude API error %s: %s", exc.status_code, exc.message)
            return AnalystResult(_no_trade(f"error API {exc.status_code}"), self.s.claude_model, error=str(exc.status_code))
        except anthropic.APIConnectionError as exc:
            log.error("Claude sin conexión: %s", exc)
            return AnalystResult(_no_trade("sin conexión con la API de Claude"), self.s.claude_model, error="connection")

        usage = response.usage
        meta = dict(
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

        if response.stop_reason == "refusal":
            log.warning("Claude rechazó el análisis: %s", getattr(response, "stop_details", None))
            return AnalystResult(_no_trade("refusal del modelo"), error="refusal", **meta)
        if response.stop_reason == "max_tokens":
            return AnalystResult(_no_trade("respuesta truncada (max_tokens)"), error="max_tokens", **meta)

        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            decision = TradeDecision.model_validate_json(text)
        except ValidationError as exc:
            log.error("Respuesta de Claude no supera la validación Pydantic: %s", exc)
            return AnalystResult(_no_trade("validación de esquema fallida"), error="validation", **meta)

        log.info(
            "Claude [%s] %s conf=%d setup=%s tokens in/out=%d/%d (cache %d)",
            response.model, decision.action.value, decision.confidence, decision.setup.value,
            meta["input_tokens"], meta["output_tokens"], meta["cache_read_tokens"],
        )
        return AnalystResult(decision, **meta)
