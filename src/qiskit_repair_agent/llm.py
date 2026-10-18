"""LLM providers. The controller depends only on the Provider protocol: generate(system, user) -> LLMResponse."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from .config import Config
from .models import LLMResponse, ProviderError

FENCE_RE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)```", re.S)


class InvalidOutput(ValueError):
    pass


def extract_code(text: str) -> str:
    """Pull the Python module out of a model reply. Prefers python-tagged fences, then any fence."""
    if not text or not text.strip():
        raise InvalidOutput("empty response")
    blocks = FENCE_RE.findall(text)
    python_blocks = [b for lang, b in blocks if lang.lower() in ("python", "py", "python3")]
    candidates = python_blocks or [b for _, b in blocks]
    if candidates:
        code = max(candidates, key=len).strip()
        if not code:
            raise InvalidOutput("code block is empty")
        return code + "\n"
    if "```" in text:
        raise InvalidOutput("unterminated code fence (response may have been cut off)")
    if re.search(r"^\s*(def |import |from \w)", text, re.M):
        return text.strip() + "\n"
    raise InvalidOutput("no Python code found in response")


class Provider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str) -> LLMResponse: ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, cfg: Config, client=None):
        import anthropic

        self._anthropic = anthropic
        self.cfg = cfg
        self.model = cfg.model
        # The SDK retries 408/409/429/5xx/connection errors with exponential backoff.
        self.client = client or anthropic.Anthropic(max_retries=cfg.provider_max_retries,
                                                    timeout=cfg.request_timeout_s)

    def generate(self, system: str, user: str) -> LLMResponse:
        a = self._anthropic
        kwargs: dict[str, Any] = dict(model=self.cfg.model, max_tokens=self.cfg.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        if self.cfg.effort:
            kwargs["output_config"] = {"effort": self.cfg.effort}
        try:
            resp = self.client.messages.create(**kwargs)
        except (a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError, a.BadRequestError) as e:
            raise ProviderError(f"{type(e).__name__} ({e.status_code}): {e.message}", fatal=True) from e
        except a.APIStatusError as e:
            raise ProviderError(f"{type(e).__name__} ({e.status_code}) after retries: {e.message}") from e
        except a.APIConnectionError as e:
            raise ProviderError(f"{type(e).__name__} after retries: {e}") from e
        except TypeError as e:   # SDK raises this when no credentials can be resolved
            raise ProviderError(f"Anthropic client misconfigured: {e}", fatal=True) from e
        if resp.stop_reason == "refusal":
            raise ProviderError("model refused the request (stop_reason=refusal)")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(text=text, model=resp.model, request_id=getattr(resp, "_request_id", None),
                           input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                           cost_usd=None,   # the Messages API does not report cost; we do not estimate it
                           stop_reason=resp.stop_reason)


class OpenRouterProvider:
    """OpenRouter chat-completions API (OpenAI-compatible), via stdlib HTTP. Needs OPENROUTER_API_KEY.
    Free (":free") models allow 20 requests/min and 50/day (1000/day after buying $10 credit)."""
    name = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"
    MIN_INTERVAL_S = 3.1          # stay under the 20 requests/minute free-tier limit
    RETRY_STATUSES = (408, 429, 500, 502, 503, 504)

    def __init__(self, cfg: Config, opener=urllib.request.urlopen, sleep=time.sleep):
        self.cfg, self.model, self._open, self._sleep = cfg, cfg.model, opener, sleep
        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self._last = 0.0

    def _fail(self, status: int | None, detail: str) -> ProviderError:
        daily = status == 429 and "per-day" in detail.lower()
        fatal = daily or status in (400, 401, 402, 403, 404)
        hint = " (free-model daily limit reached; resume tomorrow or add credits)" if daily else ""
        return ProviderError(f"OpenRouter HTTP {status}: {detail[:300]}{hint}", fatal=fatal)

    def generate(self, system: str, user: str) -> LLMResponse:
        if not self.key:
            raise ProviderError("OPENROUTER_API_KEY is not set", fatal=True)
        body = {"model": self.model, "max_tokens": self.cfg.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        data: dict[str, Any] = {}
        for attempt in range(self.cfg.provider_max_retries + 1):
            wait = self.MIN_INTERVAL_S - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                "X-Title": "qiskit-self-repair-agent"})
            try:
                with self._open(req, timeout=self.cfg.request_timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                err = self._fail(e.code, detail)
                if err.fatal or e.code not in self.RETRY_STATUSES or attempt == self.cfg.provider_max_retries:
                    raise err from e
                retry_after = e.headers.get("Retry-After") if e.headers else None
                self._sleep(float(retry_after) if retry_after and retry_after.isdigit() else 5 * 2 ** attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.cfg.provider_max_retries:
                    raise ProviderError(f"OpenRouter connection failed after retries: {e}") from e
                self._sleep(5 * 2 ** attempt)
        if "error" in data:   # errors can also arrive with HTTP 200
            err = data["error"]
            raise self._fail(err.get("code"), str(err.get("message", err)))
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return LLMResponse(text=choice["message"].get("content") or "", model=data.get("model", self.model),
                           request_id=data.get("id"), input_tokens=usage.get("prompt_tokens"),
                           output_tokens=usage.get("completion_tokens"),
                           cost_usd=usage.get("cost"),   # provider-reported; 0 for free models
                           stop_reason=choice.get("finish_reason"))


class ScriptedProvider:
    """Deterministic provider for tests: returns (or raises) the given items in order."""
    name = "mock"

    def __init__(self, responses: list, model: str = "mock-scripted"):
        self.responses = list(responses)
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        if not self.responses:
            raise ProviderError("ScriptedProvider ran out of responses", fatal=True)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item, model=self.model, input_tokens=len(user) // 4, output_tokens=len(item) // 4)


class ReferenceMockProvider:
    """Offline demo provider (`--provider mock`). Initial prompt -> a stub that raises
    NotImplementedError; repair prompt -> the task's reference solution. So baseline fails and
    repair passes on validated tasks. Exercises the full pipeline incl. the sandbox; says nothing about LLMs."""
    name = "mock"

    def __init__(self, tasks, model: str = "mock-reference"):
        self.tasks = list(tasks)
        self.model = model

    def generate(self, system: str, user: str) -> LLMResponse:
        task = next((t for t in self.tasks if t.prompt in user), None)
        if task is None:
            raise ProviderError("mock: prompt does not match any known task", fatal=True)
        if "previous solution" in user:
            code = task.reference
        else:
            code = f"def {task.entry_point}(*args, **kwargs):\n    raise NotImplementedError\n"
        return LLMResponse(text=f"```python\n{code}```", model=self.model)


def make_provider(cfg: Config, tasks=()) -> Provider:
    if cfg.provider == "mock":
        return ReferenceMockProvider(tasks)
    if cfg.provider == "openrouter":
        return OpenRouterProvider(cfg)
    return AnthropicProvider(cfg)
