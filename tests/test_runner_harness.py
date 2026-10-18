import subprocess

import pytest

from qiskit_repair_agent.config import Config
from qiskit_repair_agent.harness import classify
from qiskit_repair_agent.runner import DockerRunner, RunnerUnavailable, parse_harness_output

from conftest import make_task

# ---- harness behaviour (real harness, host subprocess, trusted test-authored code only) ----


@pytest.mark.parametrize("code,passed,category,stage", [
    ("def add(a, b):\n    return a + b\n", True, None, "test"),
    ("def add(a, b):\n    return a - b\n", False, "test_failure", "test"),
    ("def add(a, b)\n    return a + b\n", False, "syntax_error", "compile"),
    ("def add(a, b):\n    return undefined_name\n", False, "runtime_error", "test"),
    ("import qiskit_does_not_exist\n", False, "import_error", "load"),
    ("def plus(a, b):\n    return a + b\n", False, "invalid_output", "entry_point"),
    ("import sys\nsys.exit(3)\n", False, "runtime_error", "load"),
    ("def add(a, b):\n    return add(a, b)\n", False, "resource_error", "test"),
])
def test_harness_outcomes(local_runner, code, passed, category, stage):
    r = local_runner.execute(make_task(), code)
    assert (r.passed, r.category, r.stage) == (passed, category, stage), r


def test_assertion_message_reported_but_hidden_test_source_not(local_runner):
    r = local_runner.execute(make_task(), "def add(a, b):\n    return 4\n")
    assert r.message == "add(2, 3) should be 5"
    assert "<hidden_test>" in r.traceback and "assert candidate(2, 3)" not in r.traceback


def test_solution_source_visible_in_traceback(local_runner):
    r = local_runner.execute(make_task(), "def add(a, b):\n    return 1 / 0\n")
    assert r.category == "runtime_error" and "return 1 / 0" in r.traceback


def test_prelude_runs_first_and_test_sees_its_names(local_runner):
    task = make_task(prelude="from math import sqrt\n",
                     test="def check(candidate):\n    assert candidate(4, 0) == sqrt(16)\n")
    assert local_runner.execute(task, "def add(a, b):\n    return float(a + b)\n").passed


def test_tests_that_call_check_themselves_are_not_double_called(local_runner):
    task = make_task(test="CALLS = []\ndef check(candidate):\n    CALLS.append(1)\n    assert len(CALLS) == 1\n"
                          "check(add)\n")
    assert local_runner.execute(task, "def add(a, b):\n    return a + b\n").passed


def test_output_is_bounded(cfg):
    from conftest import TrustedLocalRunner
    runner = TrustedLocalRunner(Config(**{**cfg.public_dict(), "max_output_chars": 200}))
    r = runner.execute(make_task(), "import os\nprint('x' * 100000)\nos.write(2, b'y' * 100000)\n"
                                    "def add(a, b):\n    return a + b\n")
    assert r.passed and len(r.output) < 300 and "truncated" in r.output


def test_timeout_is_structured(cfg):
    from conftest import TrustedLocalRunner
    runner = TrustedLocalRunner(Config(**{**cfg.public_dict(), "exec_timeout_s": 2}))
    r = runner.execute(make_task(), "while True:\n    pass\n")
    assert (r.passed, r.category, r.timed_out) == (False, "timeout", True)


def test_classify_api_misuse():
    class QiskitError(Exception):
        pass
    QiskitError.__module__ = "qiskit.exceptions"
    assert classify(QiskitError("bad")) == "api_misuse"
    assert classify(AttributeError("'QuantumCircuit' object has no attribute 'cnot'")) == "api_misuse"
    assert classify(KeyError("x")) == "runtime_error"

# ---- runner: no host fallback, hardened command ----


def test_missing_docker_never_executes_on_host(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a))
    runner = DockerRunner(Config())
    with pytest.raises(RunnerUnavailable, match="Docker"):
        runner.check_available()
    with pytest.raises(RunnerUnavailable, match="host"):
        runner.execute(make_task(), "def add(a, b):\n    return a + b\n")
    assert called == []


def test_docker_daemon_down_is_actionable(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "docker")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "cannot connect to daemon"))
    with pytest.raises(RunnerUnavailable, match="daemon"):
        DockerRunner(Config()).check_available()


def test_docker_command_is_locked_down():
    cmd = DockerRunner(Config(memory="512m", cpus=0.5, pids_limit=64)).command("n")
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--user 10001:10001", "--cap-drop ALL",
                 "--security-opt no-new-privileges", "--memory 512m", "--memory-swap 512m", "--cpus 0.5",
                 "--pids-limit 64", "--rm"):
        assert flag in joined
    assert not any(part.startswith(("-v", "--volume", "--mount", "--privileged", "-e", "--env")) for part in cmd)
    assert "docker.sock" not in joined


def test_parse_harness_output_without_result():
    assert parse_harness_output("", "", 137, 1.0, 100).category == "resource_error"
    assert parse_harness_output("", "daemon gone", 125, 1.0, 100).category == "blocked"
    assert parse_harness_output("junk", "", 1, 1.0, 100).category == "runtime_error"
