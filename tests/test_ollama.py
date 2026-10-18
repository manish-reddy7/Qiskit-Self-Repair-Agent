import io
import json
import urllib.error

import pytest

from qiskit_repair_agent.config import Config
from qiskit_repair_agent.llm import OllamaProvider
from qiskit_repair_agent.models import ProviderError

OK = {"model": "llama3.1:latest", "message": {"role": "assistant", "content": "```python\nx = 1\n```"},
      "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 30}


def _provider(outcome, monkeypatch, **cfg):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    requests = []

    def opener(req, timeout):
        requests.append(req)
        if isinstance(outcome, Exception):
            raise outcome
        return io.BytesIO(json.dumps(outcome).encode())
    return OllamaProvider(Config(provider="ollama", model="llama3.1", **cfg), opener=opener), requests


def test_success_maps_response(monkeypatch):
    p, reqs = _provider(OK, monkeypatch, temperature=0.0)
    r = p.generate("SYS", "USER")
    assert (r.text, r.model, r.input_tokens, r.output_tokens, r.cost_usd, r.stop_reason) == \
        ("```python\nx = 1\n```", "llama3.1:latest", 100, 30, None, "stop")
    assert reqs[0].full_url == "http://localhost:11434/api/chat"
    body = json.loads(reqs[0].data)
    assert body["stream"] is False and body["options"]["temperature"] == 0.0
    assert body["options"]["num_ctx"] == OllamaProvider.NUM_CTX
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_host_from_env(monkeypatch):
    p, _ = _provider(OK, monkeypatch)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9999")
    assert OllamaProvider(Config(provider="ollama", model="m")).host == "http://127.0.0.1:9999"


def test_missing_model_is_fatal_with_pull_hint(monkeypatch):
    err = urllib.error.HTTPError("u", 404, "nf", {}, io.BytesIO(b'{"error":"model not found"}'))
    p, _ = _provider(err, monkeypatch)
    with pytest.raises(ProviderError, match="ollama pull llama3.1") as info:
        p.generate("s", "u")
    assert info.value.fatal


def test_server_down_is_fatal(monkeypatch):
    p, _ = _provider(urllib.error.URLError("refused"), monkeypatch)
    with pytest.raises(ProviderError, match="ollama serve") as info:
        p.generate("s", "u")
    assert info.value.fatal


def test_timeout_is_transient(monkeypatch):
    p, _ = _provider(TimeoutError(), monkeypatch)
    with pytest.raises(ProviderError) as info:
        p.generate("s", "u")
    assert not info.value.fatal
