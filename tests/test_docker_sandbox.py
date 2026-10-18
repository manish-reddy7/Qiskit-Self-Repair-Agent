"""Real sandbox tests. Skipped (and shown as skipped, not passed) when Docker or the image is unavailable."""
import pytest

from qiskit_repair_agent.config import Config
from qiskit_repair_agent.controller import run_task
from qiskit_repair_agent.llm import ReferenceMockProvider
from qiskit_repair_agent.runner import DockerRunner
from qiskit_repair_agent.tasks import get_task

from conftest import docker_ready

CFG = Config(provider="mock")
pytestmark = [pytest.mark.docker,
              pytest.mark.skipif(not docker_ready(CFG.image), reason=f"Docker/image {CFG.image} unavailable")]


@pytest.fixture(scope="module")
def runner():
    DockerRunner(CFG).check_available()
    return DockerRunner(CFG)


def test_reference_passes_and_stub_fails(runner):
    t = get_task("2")   # Bell-state circuit
    ok = runner.execute(t, t.reference)
    assert ok.passed and ok.qiskit_version == "2.4.2"
    bad = runner.execute(t, f"def {t.entry_point}(*a, **k):\n    return None\n")
    assert not bad.passed and bad.category in ("test_failure", "runtime_error", "api_misuse")


@pytest.mark.parametrize("snippet,needle", [
    ("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)", "unreachable"),
    ("open('/usr/local/x', 'w')", "Read-only"),
    ("import os\nassert os.getuid() != 0, 'running as root'\nraise SystemError(f'uid={os.getuid()}')", "uid=10001"),
    ("import os\nraise SystemError(sorted(k for k in os.environ if k.startswith(('ANTHROPIC', 'QRA_', 'QISKIT'))))",
     "[]"),
])
def test_sandbox_restrictions(runner, snippet, needle, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-host-secret")   # set on the host; must not reach the sandbox
    t = get_task("0")
    r = runner.execute(t, snippet + "\n")
    assert not r.passed and needle in r.message, r


def test_timeout_kills_container(runner):
    fast = DockerRunner(Config(provider="mock", exec_timeout_s=5))
    r = fast.execute(get_task("0"), "while True:\n    pass\n")
    assert r.category == "timeout" and r.timed_out


def test_memory_limit(runner):
    small = DockerRunner(Config(provider="mock", memory="256m"))
    r = small.execute(get_task("0"), "x = bytearray(1024 * 1024 * 1024)\n")
    assert not r.passed and r.category == "resource_error"


def test_end_to_end_repair_with_mock_and_real_sandbox(runner):
    t = get_task("2")
    rec = run_task(t, "repair", ReferenceMockProvider([t]), runner, CFG)
    assert (rec.initial_passed, rec.final_passed, rec.repairs_used) == (False, True, 1)
    assert rec.attempts[0].category == "runtime_error"
