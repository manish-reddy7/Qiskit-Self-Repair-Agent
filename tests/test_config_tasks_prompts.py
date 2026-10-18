import json

import pytest

from qiskit_repair_agent.config import Config, load_config
from qiskit_repair_agent.models import ExecResult
from qiskit_repair_agent.prompts import SYSTEM_PROMPT, failure_summary, initial_prompt, repair_prompt
from qiskit_repair_agent.tasks import (TaskError, get_task, load_tasks, load_tasks_from_file, normalize_task_id,
                                       select_tasks)

# ---- configuration ----


def test_defaults_are_valid():
    c = Config()
    assert c.max_repairs == 2 and c.exec_timeout_s == 30 and c.temperature is None


@pytest.mark.parametrize("kwargs", [{"max_repairs": -1}, {"provider": "openai"}, {"memory": "lots"},
                                    {"exec_timeout_s": 0}, {"temperature": 3.0}, {"image": "x; rm -rf /"},
                                    {"dataset": "nope"}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_toml_env_and_override_precedence(tmp_path, monkeypatch):
    p = tmp_path / "c.toml"
    p.write_text('[agent]\nmax_repairs = 3\nmodel = "from-file"\nexec_timeout_s = 12\n')
    monkeypatch.setenv("QRA_MODEL", "from-env")
    c = load_config(p, max_repairs=1)
    assert (c.max_repairs, c.model, c.exec_timeout_s) == (1, "from-env", 12.0)


def test_unknown_config_key_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[agent]\nmax_repairz = 3\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(p)


def test_config_never_contains_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SECRET")
    assert "SECRET" not in json.dumps(load_config().public_dict())

# ---- task registry ----


def test_vendored_benchmark_loads():
    for ds in ("standard", "hard"):
        tasks = load_tasks(ds)
        assert len(tasks) == 151
        assert len({t.task_id for t in tasks}) == 151
        assert all(t.version and t.entry_point.isidentifier() for t in tasks)


def test_task_versions_are_stable_and_content_based():
    a, b = get_task("0"), get_task("qiskitHumanEval/0")
    assert a.version == b.version and a.version != get_task("1").version


@pytest.mark.parametrize("bad", ["", "abc", "qiskitHumanEval/x", "../etc/passwd"])
def test_bad_task_ids_rejected(bad):
    with pytest.raises(TaskError):
        normalize_task_id(bad)


@pytest.mark.parametrize("mutation", [{"test": ""}, {"entry_point": "not an id"}, {"test": "assert True"},
                                      {"task_id": "other/1"}, {"prompt": None}])
def test_malformed_task_rejected(tmp_path, mutation):
    raw = {"task_id": "qiskitHumanEval/1", "prompt": "def f():\n    pass\n", "canonical_solution": "  return 1\n",
           "test": "def check(candidate):\n    assert candidate() == 1\n", "entry_point": "f"}
    raw.update(mutation)
    p = tmp_path / "t.json"
    p.write_text(json.dumps([raw]))
    with pytest.raises(TaskError):
        load_tasks_from_file(p, "standard")


def test_selection_uses_only_validated_tasks():
    tasks = select_tasks("standard")
    assert len(tasks) >= 20
    ids = {t.task_id for t in tasks}
    assert "qiskitHumanEval/43" not in ids          # needs an IBM Quantum account -> excluded by validation
    with pytest.raises(TaskError, match="not validated"):
        select_tasks("standard", ids=["43"])
    assert len(select_tasks("standard", limit=3)) == 3

# ---- prompts ----


def test_initial_prompt_has_task_not_tests():
    for ds in ("standard", "hard"):
        for t in load_tasks(ds):
            p = initial_prompt(t)
            assert t.prompt.strip() in p and t.entry_point in p
            assert "def check(" not in p and "def check(" not in SYSTEM_PROMPT


def test_repair_prompt_contains_feedback_and_code_but_not_test_source():
    t = get_task("0")
    res = ExecResult(passed=False, category="test_failure", stage="test", error_type="AssertionError",
                     message="Expected 3 qubits, got 2",
                     traceback='File "<hidden_test>", line 4, in check\nAssertionError: Expected 3 qubits, got 2',
                     output="hello")
    summary = failure_summary("CODE", res, None, 4000)
    p = repair_prompt(t, "def create_quantum_circuit(n):\n    pass\n", summary)
    assert "Expected 3 qubits, got 2" in p and "test_failure" in p and "def create_quantum_circuit" in p
    assert "def check(" not in p and t.test.strip() not in p


def test_failure_summary_variants_and_truncation():
    assert "invalid_output" in failure_summary(None, None, "no Python code found", 100)
    timeout = ExecResult(passed=False, category="timeout", message="execution exceeded 30s and was killed")
    assert "exceeded 30s" in failure_summary("x", timeout, None, 100)
    big = ExecResult(passed=False, category="runtime_error", stage="load", error_type="ValueError", message="m",
                     traceback="T" * 10_000, output="O" * 10_000)
    assert len(failure_summary("x", big, None, 500)) < 1200
