"""Settings: defaults < TOML file ([agent] table) < QRA_* environment variables < CLI flags.

Secrets are never part of Config; the Anthropic SDK reads ANTHROPIC_API_KEY itself.
"""
from __future__ import annotations

import dataclasses
import os
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DATASETS = ("standard", "hard")
PROVIDERS = ("anthropic", "openrouter", "ollama", "mock")


@dataclass(frozen=True)
class Config:
    provider: str = "anthropic"
    model: str = "claude-opus-5-5"
    temperature: float | None = None     # None = not sent (Opus 5.5 rejects sampling parameters)
    effort: str | None = "medium"        # output_config.effort; None = not sent
    max_tokens: int = 16000
    request_timeout_s: float = 300.0
    provider_max_retries: int = 3        # SDK-level retries with exponential backoff (429/5xx/network)
    max_repairs: int = 2                 # additional attempts after the initial generation
    max_run_seconds: float = 900.0       # wall-clock budget for one task run (all attempts)
    max_run_tokens: int = 200_000        # input+output token budget for one task run
    exec_timeout_s: float = 30.0
    memory: str = "1g"
    cpus: float = 1.0
    pids_limit: int = 256
    max_output_chars: int = 4000         # cap on captured program output / tracebacks / feedback
    image: str = "qiskit-repair-runner:2.4.2"
    dataset: str = "standard"
    results_dir: str = "results"

    def __post_init__(self):
        errors = []
        if self.provider not in PROVIDERS:
            errors.append(f"provider must be one of {PROVIDERS}")
        if self.dataset not in DATASETS:
            errors.append(f"dataset must be one of {DATASETS}")
        if not self.model.strip():
            errors.append("model must not be empty")
        if self.provider == "openrouter" and "/" not in self.model:
            errors.append("openrouter needs a model id like 'nvidia/nemotron-3-super-120b-a12b:free' (--model)")
        if self.temperature is not None and not 0 <= self.temperature <= 1:
            errors.append("temperature must be in [0, 1]")
        if self.effort not in (None, "low", "medium", "high", "xhigh", "max"):
            errors.append("effort must be low|medium|high|xhigh|max or unset")
        if not 0 <= self.max_repairs <= 10:
            errors.append("max_repairs must be in [0, 10]")
        for name in ("max_tokens", "request_timeout_s", "max_run_seconds", "max_run_tokens",
                     "exec_timeout_s", "cpus", "pids_limit", "max_output_chars"):
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be > 0")
        if self.provider_max_retries < 0:
            errors.append("provider_max_retries must be >= 0")
        if not re.fullmatch(r"\d+[kmg]", self.memory):
            errors.append("memory must look like 512m or 1g")
        if not re.fullmatch(r"[\w./:-]+", self.image):
            errors.append("image must be a plain docker image reference")
        if errors:
            raise ValueError("Invalid configuration: " + "; ".join(errors))

    def public_dict(self) -> dict:
        return dataclasses.asdict(self)


def _cast(raw: str, default):
    if raw.lower() in ("", "none", "null"):
        return None
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float) or default is None:   # only temperature defaults to None
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env support: KEY=VALUE lines, never overriding variables already set."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_config(path: str | Path | None = None, **overrides) -> Config:
    values: dict = {}
    if path:
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"Config file not found: {p}")
        values.update(tomllib.loads(p.read_text(encoding="utf-8")).get("agent", {}))
    known = {f.name: f.default for f in fields(Config)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for name, default in known.items():
        raw = os.environ.get(f"QRA_{name.upper()}")
        if raw is not None:
            values[name] = _cast(raw, default)
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**values)
