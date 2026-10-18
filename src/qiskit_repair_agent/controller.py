"""Baseline and self-repair loops. Both modes share the same initial prompt and generation settings;
the only difference is that repair mode may feed failures back for up to cfg.max_repairs more attempts."""
from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, UTC

from .config import Config
from .llm import InvalidOutput, Provider, extract_code
from .models import Attempt, ExecResult, ProviderError, RunRecord, Task
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, failure_summary, initial_prompt, repair_prompt
from .runner import RunnerUnavailable

MODES = ("baseline", "repair")


def _attempt(index: int, retry_reason: str | None, user: str, task: Task, provider: Provider, runner) -> Attempt:
    start = time.monotonic()
    att = Attempt(index=index, kind="initial" if index == 0 else "repair", retry_reason=retry_reason,
                  passed=False, category=None, code=None, code_sha256=None)
    try:
        resp = provider.generate(SYSTEM_PROMPT, user)
    except ProviderError as e:
        if e.fatal:
            raise
        att.category, att.llm_error = "provider_error", str(e)
        att.llm_latency_s = att.duration_s = time.monotonic() - start
        return att
    att.llm_latency_s = time.monotonic() - start
    att.llm_model, att.llm_request_id = resp.model, resp.request_id
    att.input_tokens, att.output_tokens, att.cost_usd = resp.input_tokens, resp.output_tokens, resp.cost_usd
    try:
        att.code = extract_code(resp.text)
    except InvalidOutput as e:
        att.category = "invalid_output"
        att.llm_error = f"{e} (stop_reason={resp.stop_reason})"
    else:
        att.code_sha256 = hashlib.sha256(att.code.encode()).hexdigest()
        try:
            att.execution = runner.execute(task, att.code)
        except RunnerUnavailable as e:
            att.execution = ExecResult(passed=False, category="blocked", stage="sandbox",
                                       error_type="RunnerUnavailable", message=str(e))
        att.passed, att.category = att.execution.passed, att.execution.category
    att.duration_s = time.monotonic() - start
    return att


def _sum(values):
    values = list(values)
    return None if not values or any(v is None for v in values) else sum(values)


def run_task(task: Task, mode: str, provider: Provider, runner, cfg: Config,
             eval_id: str = "adhoc", repeat: int = 0) -> RunRecord:
    """Run one task. Raises ProviderError only for fatal provider errors (auth, bad model)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    start = time.monotonic()
    attempts: list[Attempt] = []
    user, retry_reason = initial_prompt(task), None
    while True:
        att = _attempt(len(attempts), retry_reason, user, task, provider, runner)
        attempts.append(att)
        tokens = sum((a.input_tokens or 0) + (a.output_tokens or 0) for a in attempts)
        if att.passed:
            stop = "passed"
        elif att.category == "provider_error":
            stop = "provider_error"
        elif att.category == "blocked":
            stop = "blocked"
        elif mode == "baseline":
            stop = "baseline_single_attempt"
        elif len(attempts) - 1 >= cfg.max_repairs:
            stop = "max_repairs_reached"
        elif time.monotonic() - start >= cfg.max_run_seconds:
            stop = "time_budget_exhausted"
        elif tokens >= cfg.max_run_tokens:
            stop = "token_budget_exhausted"
        else:
            summary = failure_summary(att.code, att.execution, att.llm_error, cfg.max_output_chars)
            user, retry_reason = repair_prompt(task, att.code, summary), att.category
            continue
        break

    last = attempts[-1]
    if last.passed:
        status = "passed"
    elif last.category in ("provider_error", "blocked"):
        status = last.category
    elif last.category == "timeout":
        status = "timed_out"
    else:
        status = "failed"
    return RunRecord(
        run_id=uuid.uuid4().hex, eval_id=eval_id, task_id=task.task_id, task_version=task.version,
        dataset=task.source, mode=mode, repeat=repeat, model_id=provider.model, provider=provider.name,
        prompt_version=PROMPT_VERSION, timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        final_status=status, stop_reason=stop, initial_passed=attempts[0].passed, final_passed=last.passed,
        repairs_used=len(attempts) - 1, total_latency_s=time.monotonic() - start,
        input_tokens=_sum(a.input_tokens for a in attempts if a.category != "provider_error"),
        output_tokens=_sum(a.output_tokens for a in attempts if a.category != "provider_error"),
        cost_usd=_sum(a.cost_usd for a in attempts if a.category != "provider_error"),
        attempts=attempts,
    )
