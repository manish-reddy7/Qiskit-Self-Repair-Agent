import subprocess
import sys

import pytest

from qiskit_repair_agent.config import Config
from qiskit_repair_agent.models import ExecResult, Task
from qiskit_repair_agent.runner import HARNESS_SOURCE, DockerRunner


class TrustedLocalRunner(DockerRunner):
    """TEST ONLY: runs the real harness in a host subprocess. Used exclusively with code written in
    these tests (never LLM output), so harness/timeout logic can be tested without Docker."""

    def _preflight(self):
        pass

    def command(self, name):
        return [sys.executable, "-I", "-c", HARNESS_SOURCE]

    def _kill(self, name):
        pass


class FakeRunner:
    """Scripted runner for controller tests: passes iff the code contains 'GOOD'."""

    def __init__(self, fail_category="test_failure"):
        self.calls = []
        self.fail_category = fail_category

    def check_available(self):
        return {"image_id": "fake"}

    def execute(self, task, code):
        self.calls.append(code)
        if "GOOD" in code:
            return ExecResult(passed=True, category=None, stage="test", duration_s=0.01)
        return ExecResult(passed=False, category=self.fail_category, stage="test", error_type="AssertionError",
                          message="expected 5, got 4", duration_s=0.01)


def make_task(test="def check(candidate):\n    assert candidate(2, 3) == 5, 'add(2, 3) should be 5'\n",
              entry_point="add", prelude="", task_id="qiskitHumanEval/900"):
    return Task(task_id=task_id, version="v-test", prompt=f"def {entry_point}(a, b):\n    \"\"\"Add.\"\"\"\n",
                prelude=prelude, test=test, entry_point=entry_point,
                reference=f"def {entry_point}(a, b):\n    return a + b\n", difficulty="basic", source="unit")


@pytest.fixture
def cfg(tmp_path):
    return Config(provider="mock", model="mock", exec_timeout_s=10, results_dir=str(tmp_path / "results"))


@pytest.fixture
def local_runner(cfg):
    return TrustedLocalRunner(cfg)


def docker_ready(image):
    try:
        ok = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
    return ok
