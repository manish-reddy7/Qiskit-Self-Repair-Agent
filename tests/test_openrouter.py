import io
import json
import urllib.error

import pytest

from qiskit_repair_agent.config import Config
from qiskit_repair_agent.llm import OpenRouterProvider
from qiskit_repair_agent.models import ProviderError

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


class FakeOpener:
    """Replays scripted HTTP outcomes: dict -> 200 JSON body, (status, body) -> HTTPError."""

    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def __call__(self, req, timeout):
        self.requests.append(req)
        out = self.outcomes.pop(0)
        if isinstance(out, tuple):
            status, body = out
            raise urllib.error.HTTPError(req.full_url, status, "err", {}, io.BytesIO(body.encode()))
        return io.BytesIO(json.dumps(out).encode())


OK = {"id": "gen-1", "model": MODEL, "choices": [{"message": {"content": "```python\nx = 1\n```"},
                                                   "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 120, "completion_tokens": 40, "cost": 0}}


def _provider(monkeypatch, outcomes, key="sk-or-test", **cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    sleeps = []
    opener = FakeOpener(outcomes)
    p = OpenRouterProvider(Config(provider="openrouter", model=MODEL, **cfg), opener=opener, sleep=sleeps.append)
    return p, opener, sleeps


def test_success_maps_text_usage_and_reported_cost(monkeypatch):
    p, opener, _ = _provider(monkeypatch, [OK])
    r = p.generate("SYS", "USER")
    assert (r.text, r.model, r.request_id, r.input_tokens, r.output_tokens, r.cost_usd, r.stop_reason) == \
        ("```python\nx = 1\n```", MODEL, "gen-1", 120, 40, 0, "stop")
    body = json.loads(opener.requests[0].data)
    assert body["model"] == MODEL and [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "temperature" not in body
    assert opener.requests[0].get_header("Authorization") == "Bearer sk-or-test"


def test_temperature_sent_when_configured(monkeypatch):
    p, opener, _ = _provider(monkeypatch, [OK], temperature=0.0)
    p.generate("s", "u")
    assert json.loads(opener.requests[0].data)["temperature"] == 0.0


def test_rate_limit_retried_then_succeeds(monkeypatch):
    p, opener, sleeps = _provider(monkeypatch, [(429, "Rate limit exceeded: free-models-per-min"), OK])
    assert p.generate("s", "u").text
    assert len(opener.requests) == 2 and any(s >= 5 for s in sleeps)


def test_requests_are_paced_for_free_tier(monkeypatch):
    p, _, sleeps = _provider(monkeypatch, [OK, OK])
    p.generate("s", "u")
    p.generate("s", "u")
    assert sleeps and 0 < sleeps[-1] <= OpenRouterProvider.MIN_INTERVAL_S


@pytest.mark.parametrize("status,body", [(401, "No auth credentials found"), (402, "Insufficient credits"),
                                         (404, "No endpoints found"),
                                         (429, "Rate limit exceeded: free-models-per-day")])
def test_non_retryable_errors_are_fatal(monkeypatch, status, body):
    p, opener, _ = _provider(monkeypatch, [(status, body)])
    with pytest.raises(ProviderError) as info:
        p.generate("s", "u")
    assert info.value.fatal and len(opener.requests) == 1


def test_persistent_server_error_is_transient_provider_error(monkeypatch):
    p, opener, _ = _provider(monkeypatch, [(503, "busy")] * 4, provider_max_retries=3)
    with pytest.raises(ProviderError) as info:
        p.generate("s", "u")
    assert not info.value.fatal and len(opener.requests) == 4


def test_error_inside_200_body(monkeypatch):
    p, _, _ = _provider(monkeypatch, [{"error": {"code": 502, "message": "upstream failed"}}])
    with pytest.raises(ProviderError, match="upstream failed"):
        p.generate("s", "u")


def test_missing_key_is_fatal(monkeypatch):
    p, opener, _ = _provider(monkeypatch, [OK], key="")
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY") as info:
        p.generate("s", "u")
    assert info.value.fatal and opener.requests == []


def test_openrouter_requires_namespaced_model():
    with pytest.raises(ValueError, match="openrouter"):
        Config(provider="openrouter")      # default model is an Anthropic id
