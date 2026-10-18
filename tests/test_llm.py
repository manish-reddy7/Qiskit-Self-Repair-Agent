from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from qiskit_repair_agent.config import Config
from qiskit_repair_agent.llm import (AnthropicProvider, InvalidOutput, ReferenceMockProvider, ScriptedProvider,
                                     extract_code)
from qiskit_repair_agent.models import ProviderError
from qiskit_repair_agent.prompts import initial_prompt, repair_prompt
from qiskit_repair_agent.tasks import get_task


@pytest.mark.parametrize("text,expected", [
    ("```python\ndef f():\n    return 1\n```", "def f():\n    return 1\n"),
    ("Here you go:\n```py\nimport x\n```\nThanks", "import x\n"),
    ("```\nfrom qiskit import QuantumCircuit\n```", "from qiskit import QuantumCircuit\n"),
    ("```bash\npip install\n```\n```python\nimport qiskit\nX = 1\n```", "import qiskit\nX = 1\n"),
    ("def f():\n    return 2", "def f():\n    return 2\n"),
])
def test_extract_code(text, expected):
    assert extract_code(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "I cannot help with that.", "```python\n\n```",
                                  "```python\ndef f():\n    return 1\n"])
def test_extract_code_rejects_unusable(text):
    with pytest.raises(InvalidOutput):
        extract_code(text)


def test_scripted_provider_order_and_errors():
    p = ScriptedProvider(["a", ProviderError("boom")])
    assert p.generate("s", "u").text == "a"
    with pytest.raises(ProviderError):
        p.generate("s", "u")
    with pytest.raises(ProviderError):   # exhausted
        p.generate("s", "u")
    assert len(p.calls) == 3


def test_reference_mock_fails_first_then_repairs():
    t = get_task("0")
    p = ReferenceMockProvider([t])
    first = extract_code(p.generate("s", initial_prompt(t)).text)
    assert "NotImplementedError" in first
    second = extract_code(p.generate("s", repair_prompt(t, first, "err")).text)
    assert second == t.reference


class _FakeMessages:
    def __init__(self, result):
        self.result, self.kwargs = result, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _provider(result, **cfg):
    msgs = _FakeMessages(result)
    return AnthropicProvider(Config(**cfg), client=SimpleNamespace(messages=msgs)), msgs


def _resp(text="```python\nx=1\n```", stop="end_turn"):
    blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
    r = SimpleNamespace(content=blocks,
                        model="claude-opus-5-5", stop_reason=stop,
                        usage=SimpleNamespace(input_tokens=11, output_tokens=22))
    r._request_id = "req_123"
    return r


def test_anthropic_provider_maps_response_and_usage():
    p, msgs = _provider(_resp())
    r = p.generate("SYS", "USER")
    assert (r.text, r.model, r.request_id, r.input_tokens, r.output_tokens) == \
        ("```python\nx=1\n```", "claude-opus-5-5", "req_123", 11, 22)
    assert r.cost_usd is None                                  # never fabricated
    assert msgs.kwargs["system"] == "SYS" and msgs.kwargs["messages"][0]["content"] == "USER"
    assert "temperature" not in msgs.kwargs                    # Opus 5.5 rejects sampling params
    assert msgs.kwargs["output_config"] == {"effort": "medium"}


def test_anthropic_provider_sends_temperature_only_when_configured():
    p, msgs = _provider(_resp(), temperature=0.0, effort=None)
    p.generate("s", "u")
    assert msgs.kwargs["temperature"] == 0.0 and "output_config" not in msgs.kwargs


def _status_error(cls, code):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("err", response=httpx.Response(code, request=req), body=None)


@pytest.mark.parametrize("exc,fatal", [
    (_status_error(anthropic.AuthenticationError, 401), True),
    (_status_error(anthropic.NotFoundError, 404), True),
    (_status_error(anthropic.RateLimitError, 429), False),
    (_status_error(anthropic.InternalServerError, 500), False),
    (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), False),
])
def test_anthropic_errors_become_provider_errors(exc, fatal):
    p, _ = _provider(exc)
    with pytest.raises(ProviderError) as info:
        p.generate("s", "u")
    assert info.value.fatal is fatal


def test_refusal_is_provider_error_not_code():
    p, _ = _provider(_resp(text="", stop="refusal"))
    with pytest.raises(ProviderError, match="refus"):
        p.generate("s", "u")
