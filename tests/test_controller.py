import pytest

from qiskit_repair_agent.config import Config
from qiskit_repair_agent.controller import run_task
from qiskit_repair_agent.llm import ScriptedProvider
from qiskit_repair_agent.models import ProviderError
from qiskit_repair_agent.runner import RunnerUnavailable

from conftest import FakeRunner, make_task

BAD = "```python\ndef add(a, b):\n    return a - b  # BAD\n```"
GOOD = "```python\ndef add(a, b):\n    return a + b  # GOOD\n```"


def _run(responses, mode, cfg=None, runner=None):
    provider, runner = ScriptedProvider(responses), runner or FakeRunner()
    rec = run_task(make_task(), mode, provider, runner, cfg or Config(provider="mock"))
    return rec, provider, runner


def test_baseline_stops_after_one_attempt_even_on_failure():
    rec, provider, runner = _run([BAD, GOOD], "baseline")
    assert len(provider.calls) == 1 and len(runner.calls) == 1
    assert (rec.final_status, rec.stop_reason, rec.repairs_used) == ("failed", "baseline_single_attempt", 0)


def test_repair_stops_on_first_success():
    rec, provider, _ = _run([BAD, GOOD, BAD], "repair")
    assert len(provider.calls) == 2
    assert (rec.final_status, rec.stop_reason, rec.repairs_used) == ("passed", "passed", 1)
    assert not rec.initial_passed and rec.final_passed
    assert [a.kind for a in rec.attempts] == ["initial", "repair"]
    assert rec.attempts[1].retry_reason == "test_failure"


def test_repair_respects_retry_limit():
    rec, provider, _ = _run([BAD] * 10, "repair", Config(provider="mock", max_repairs=2))
    assert len(provider.calls) == 3      # initial + 2 repairs
    assert (rec.final_status, rec.stop_reason, rec.repairs_used) == ("failed", "max_repairs_reached", 2)


def test_zero_repairs_makes_repair_mode_one_shot():
    rec, provider, _ = _run([BAD, GOOD], "repair", Config(provider="mock", max_repairs=0))
    assert len(provider.calls) == 1 and rec.stop_reason == "max_repairs_reached"


def test_initial_prompt_identical_across_modes():
    _, p1, _ = _run([BAD], "baseline")
    _, p2, _ = _run([BAD, GOOD], "repair")
    assert p1.calls[0] == p2.calls[0]


def test_repair_prompt_carries_previous_code_and_error():
    _, provider, _ = _run([BAD, GOOD], "repair")
    second = provider.calls[1][1]
    assert "return a - b" in second and "expected 5, got 4" in second and "def check(" not in second


def test_passing_first_attempt_needs_no_repair():
    rec, provider, _ = _run([GOOD], "repair")
    assert rec.initial_passed and rec.repairs_used == 0 and len(provider.calls) == 1


def test_invalid_output_is_not_executed_and_is_repaired():
    rec, _, runner = _run(["Sorry, no code.", GOOD], "repair")
    assert rec.attempts[0].category == "invalid_output" and rec.attempts[0].execution is None
    assert len(runner.calls) == 1 and rec.final_status == "passed"


def test_transient_provider_error_ends_run_without_fake_code():
    rec, _, runner = _run([ProviderError("overloaded")], "repair")
    assert rec.final_status == "provider_error" and runner.calls == []
    assert rec.attempts[0].code is None and rec.input_tokens is None


def test_fatal_provider_error_propagates():
    with pytest.raises(ProviderError):
        _run([ProviderError("bad key", fatal=True)], "repair")


def test_token_budget_stops_loop():
    rec, provider, _ = _run([BAD] * 5, "repair", Config(provider="mock", max_repairs=5, max_run_tokens=1))
    assert rec.stop_reason == "token_budget_exhausted" and len(provider.calls) == 1


def test_time_budget_stops_loop(monkeypatch):
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr("qiskit_repair_agent.controller.time.monotonic", lambda: next(clock))
    rec, provider, _ = _run([BAD] * 5, "repair", Config(provider="mock", max_repairs=5, max_run_seconds=150))
    assert rec.stop_reason == "time_budget_exhausted" and len(provider.calls) == 1


def test_timeout_final_status():
    rec, _, _ = _run([BAD], "baseline", runner=FakeRunner(fail_category="timeout"))
    assert rec.final_status == "timed_out"


def test_blocked_sandbox_stops_immediately():
    class Blocked(FakeRunner):
        def execute(self, task, code):
            raise RunnerUnavailable("no docker")
    rec, provider, _ = _run([BAD, GOOD], "repair", runner=Blocked())
    assert rec.final_status == "blocked" and len(provider.calls) == 1


def test_every_attempt_traced():
    rec, _, _ = _run([BAD, BAD, GOOD], "repair")
    assert [a.index for a in rec.attempts] == [0, 1, 2]
    assert all(a.code and a.code_sha256 and a.execution for a in rec.attempts)
    assert rec.input_tokens == sum(a.input_tokens for a in rec.attempts)
