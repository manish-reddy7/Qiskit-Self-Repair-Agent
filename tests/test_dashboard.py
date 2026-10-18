from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from qiskit_repair_agent.evaluator import evaluate  # noqa: E402
from qiskit_repair_agent.llm import ScriptedProvider  # noqa: E402
from qiskit_repair_agent.reporting import load_records  # noqa: E402

from conftest import FakeRunner, make_task  # noqa: E402

APP = str(Path(__file__).parents[1] / "dashboard" / "app.py")


def test_dashboard_renders_an_evaluation(cfg, monkeypatch):
    tasks = [make_task(task_id="qiskitHumanEval/1"), make_task(task_id="qiskitHumanEval/2")]
    bad = "```python\ndef add(a, b):\n    return 0\n```"
    good = "```python\n# GOOD\ndef add(a, b):\n    return a + b\n```"
    evaluate(tasks, ["baseline", "repair"], ScriptedProvider([bad, bad, good, good, good]), FakeRunner(), cfg,
             log=lambda *_: None)
    monkeypatch.setattr("sys.argv", ["app.py", cfg.results_dir])
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Repair final pass" == m.label for m in at.metric)
    assert "return 0" in at.code[0].value            # first run = failing baseline for task 1
    out = next(Path(cfg.results_dir).iterdir())
    repair = next(r for r in load_records(out / "runs.jsonl") if r.mode == "repair" and r.task_id.endswith("/1"))
    at = _select(at, repair.run_id)
    assert not at.exception
    assert any("return a + b" in c.value for c in at.code)  # repair trace shows the fixed attempt


def _select(at, run_id):
    run_box = next(s for s in at.selectbox if s.label == "Run")
    return run_box.set_value(run_id).run()


def test_dashboard_handles_no_results(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["app.py", str(tmp_path)])
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and "No results" in at.info[0].value
