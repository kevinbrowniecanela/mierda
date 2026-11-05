import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from brain.claude_analyst import ClaudeAnalyst
from brain.schemas import Action, TradeDecision


def test_schema_requires_stop_for_trades():
    with pytest.raises(ValidationError):
        TradeDecision(action="LONG", setup="breakout", confidence=80, stop_loss=None, thesis="x", invalidation="y")
    with pytest.raises(ValidationError):
        TradeDecision.model_validate_json('{"action":"LONG","setup":"breakout","confidence":150,"stop_loss":1,'
                                          '"thesis":"x","invalidation":"y","risk_flags":[]}')
    d = TradeDecision(action="SHORT", setup="mean_reversion", confidence=75, stop_loss=110, thesis="x", invalidation="y")
    assert d.stop_is_coherent(100) and not d.stop_is_coherent(120)


class FakeBetaMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.kwargs = payload, stop_reason, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            model=kwargs["model"], stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=self.payload)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
        )


@pytest.mark.asyncio
async def test_analyst_parses_and_sends_schema(settings):
    payload = json.dumps({"action": "LONG", "setup": "breakout", "confidence": 82, "stop_loss": 97.5,
                          "thesis": "ruptura con OI", "invalidation": "cierre bajo 97.5", "risk_flags": []})
    analyst = ClaudeAnalyst(settings)
    fake = FakeBetaMessages(payload)
    analyst.client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    res = await analyst.analyze({"coin": "BTC"}, ["- titular"])
    assert res.decision.action == Action.LONG and res.decision.confidence == 82
    assert fake.kwargs["output_config"]["format"]["type"] == "json_schema"
    assert fake.kwargs["fallbacks"] == "default"


@pytest.mark.asyncio
async def test_analyst_invalid_or_refusal_means_no_trade(settings):
    analyst = ClaudeAnalyst(settings)
    analyst.client = SimpleNamespace(beta=SimpleNamespace(messages=FakeBetaMessages("{not json")))
    assert (await analyst.analyze({}, [])).decision.action == Action.NO_TRADE
    analyst.client = SimpleNamespace(beta=SimpleNamespace(messages=FakeBetaMessages("", stop_reason="refusal")))
    res = await analyst.analyze({}, [])
    assert res.decision.action == Action.NO_TRADE and res.error == "refusal"
