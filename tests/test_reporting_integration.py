import csv
import json

import pytest

from qiskit_repair_agent.config import Config
from qiskit_repair_agent.controller import run_task
from qiskit_repair_agent.evaluator import evaluate
from qiskit_repair_agent.llm import ScriptedProvider
from qiskit_repair_agent.models import Attempt, ProviderError, RunRecord
from qiskit_repair_agent.reporting import append_record, compute_metrics, load_records, write_exports

from conftest import FakeRunner, TrustedLocalRunner, make_task


def _rec(task, mode, initial, final, repairs, status=None, latency=1.0, tokens=10, cost=None, cats=()):
    attempts = [Attempt(index=i, kind="initial" if i == 0 else "repair", retry_reason=None,
                        passed=(i == repairs and final), category=None if (i == repairs and final) else
                        (cats[i] if i < len(cats) else "test_failure"), code="x", code_sha256="h",
                        input_tokens=tokens, output_tokens=tokens)
                for i in range(repairs + 1)]
    return RunRecord(run_id=f"{task}{mode}", eval_id="e", task_id=f"qiskitHumanEval/{task}", task_version="v",
                     dataset="d", mode=mode, repeat=0, model_id="m", provider="mock", prompt_version="v1",
                     timestamp="t", final_status=status or ("passed" if final else "failed"), stop_reason="s",
                     initial_passed=initial, final_passed=final, repairs_used=repairs, total_latency_s=latency,
                     input_tokens=tokens * (repairs + 1), output_tokens=tokens * (repairs + 1), cost_usd=cost,
                     attempts=attempts)


@pytest.fixture
def records():
    return [
        _rec(1, "baseline", True, True, 0), _rec(1, "repair", True, True, 0, latency=1.0),
        _rec(2, "baseline", False, False, 0), _rec(2, "repair", False, True, 1, latency=3.0),
        _rec(3, "baseline", False, False, 0), _rec(3, "repair", False, False, 2, latency=5.0,
                                                    cats=("syntax_error", "api_misuse", "test_failure")),
        _rec(4, "baseline", False, False, 0, status="provider_error"),
        _rec(4, "repair", False, False, 0, status="provider_error"),
    ]


def test_metrics(records):
    m = compute_metrics(records)
    b, r = m["modes"]["baseline"], m["modes"]["repair"]
    assert (b["n_runs"], b["n_valid"]) == (4, 3)
    assert b["first_attempt_pass_rate"] == 0.25 and b["final_pass_rate"] == 0.25
    assert b["final_pass_rate_valid"] == pytest.approx(1 / 3)
    assert r["final_pass_rate"] == 0.5 and r["first_attempt_pass_rate"] == 0.25
    assert r["n_initially_failing"] == 2 and r["repair_success_rate"] == 0.5   # task 4 excluded: provider error
    assert r["repairs_distribution"] == {"0": 2, "1": 1, "2": 1} and r["mean_repairs"] == 0.75
    assert r["latency_median_s"] == 2.0 and r["latency_mean_s"] == 2.5
    assert r["cost_usd_total"] is None                                         # unknown, not zero
    assert r["failure_distribution_all_attempts"] == {"test_failure": 3, "syntax_error": 1, "api_misuse": 1}
    c = m["comparison"]
    assert c["absolute_improvement_pp"] == 25.0 and c["initial_pass_rate_difference_pp"] == 0.0
    assert c["paired"] == {"both_pass": 1, "repair_only": 1, "baseline_only": 0, "neither": 2}


def test_known_cost_is_summed():
    m = compute_metrics([_rec(1, "repair", True, True, 0, cost=0.5), _rec(2, "repair", True, True, 0, cost=0.25)])
    assert m["modes"]["repair"]["cost_usd_total"] == 0.75


def test_jsonl_roundtrip_and_exports(tmp_path, records):
    path = tmp_path / "runs.jsonl"
    for r in records:
        append_record(path, r)
    loaded = load_records(path)
    assert loaded == records
    write_exports(loaded, tmp_path, {"eval_id": "e", "model": "m"})
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["metrics"]["comparison"]["absolute_improvement_pp"] == 25.0
    assert len(summary["per_task"]) == 8
    rows = list(csv.DictReader(open(tmp_path / "runs.csv", encoding="utf-8")))
    assert len(rows) == 8 and {r["task_id"] for r in rows} == {f"qiskitHumanEval/{i}" for i in (1, 2, 3, 4)}
    attempts = list(csv.DictReader(open(tmp_path / "attempts.csv", encoding="utf-8")))
    assert len(attempts) == sum(len(r.attempts) for r in records)
    assert "Absolute improvement" in (tmp_path / "report.md").read_text()


def test_offline_end_to_end_with_real_harness(cfg):
    """Mock LLM + real harness (host subprocess, test-authored code): baseline fails, repair fixes it."""
    tasks = [make_task(task_id="qiskitHumanEval/901"), make_task(task_id="qiskitHumanEval/902")]
    bad = "```python\ndef add(a, b):\n    return a * b\n```"
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    # per task: baseline(bad), repair(bad -> good)
    provider = ScriptedProvider([bad, bad, good, good, good])
    out, metrics = evaluate(tasks, ["baseline", "repair"], provider, TrustedLocalRunner(cfg), cfg, log=lambda *_: None)
    recs = load_records(out / "runs.jsonl")
    assert [(r.task_id[-3:], r.mode, r.final_status, r.repairs_used) for r in recs] == [
        ("901", "baseline", "failed", 0), ("901", "repair", "passed", 1),
        ("902", "baseline", "passed", 0), ("902", "repair", "passed", 0)]
    assert recs[1].attempts[0].execution.message == "add(2, 3) should be 5"
    assert metrics["comparison"]["paired"]["repair_only"] == 1
    assert {p.name for p in out.iterdir()} >= {"metadata.json", "runs.jsonl", "summary.json", "runs.csv",
                                               "attempts.csv", "report.md"}
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["tasks"] == {t.task_id: t.version for t in tasks} and meta["max_repairs"] == 2


def test_evaluation_survives_task_crash_and_saves_incrementally(cfg):
    class Exploding(FakeRunner):
        def execute(self, task, code):
            if task.task_id.endswith("1"):
                raise RuntimeError("runner bug")
            return super().execute(task, code)
    tasks = [make_task(task_id="qiskitHumanEval/1"), make_task(task_id="qiskitHumanEval/2")]
    good = "```python\n# GOOD\ndef add(a, b):\n    return a + b\n```"
    out, m = evaluate(tasks, ["baseline"], ScriptedProvider([good, good]), Exploding(), cfg, log=lambda *_: None)
    statuses = [r.final_status for r in load_records(out / "runs.jsonl")]
    assert statuses == ["internal_error", "passed"] and m["modes"]["baseline"]["n_valid"] == 1


def test_fatal_provider_error_stops_but_keeps_partial_results(cfg):
    tasks = [make_task(task_id="qiskitHumanEval/1"), make_task(task_id="qiskitHumanEval/2")]
    good = "```python\n# GOOD\ndef add(a, b):\n    return a + b\n```"
    provider = ScriptedProvider([good, ProviderError("invalid api key", fatal=True)])
    out, _ = evaluate(tasks, ["baseline"], provider, FakeRunner(), cfg, log=lambda *_: None)
    assert len(load_records(out / "runs.jsonl")) == 1
    assert "invalid api key" in json.loads((out / "metadata.json").read_text())["aborted"]


def test_run_record_has_reproducibility_fields(cfg):
    rec = run_task(make_task(), "repair", ScriptedProvider(["```python\n# GOOD\nx=1\n```"]), FakeRunner(), cfg)
    for field in ("run_id", "task_id", "task_version", "mode", "model_id", "prompt_version", "timestamp",
                  "final_status", "total_latency_s"):
        assert getattr(rec, field) not in (None, "")


def test_mcnemar_exact():
    from qiskit_repair_agent.reporting import mcnemar_exact_p
    assert mcnemar_exact_p(0, 0) is None
    assert mcnemar_exact_p(5, 0) == pytest.approx(2 / 32)          # 2 * 0.5**5
    assert mcnemar_exact_p(3, 3) == 1.0
    assert mcnemar_exact_p(10, 2) == pytest.approx(0.03857, abs=1e-4)


def test_comparison_reports_significance_fields(records):
    c = compute_metrics(records)["comparison"]
    assert c["mcnemar_exact_p"] == 1.0 and c["mcnemar_pairs_independent"] is True   # 1 discordant pair
    assert c["improvement_pp_by_repeat"] == [25.0] and c["improvement_pp_stdev"] is None


def test_resume_skips_finished_runs_and_reuses_settings(cfg):
    from qiskit_repair_agent.evaluator import load_resume
    tasks = [make_task(task_id="qiskitHumanEval/1"), make_task(task_id="qiskitHumanEval/2")]
    good = "```python\n# GOOD\ndef add(a, b):\n    return a + b\n```"
    out, _ = evaluate(tasks, ["baseline"], ScriptedProvider([good, ProviderError("daily limit", fatal=True)]),
                      FakeRunner(), cfg, log=lambda *_: None)
    assert len(load_records(out / "runs.jsonl")) == 1
    meta = json.loads((out / "metadata.json").read_text())
    provider = ScriptedProvider([good])
    out2, m = evaluate(tasks, ["baseline"], provider, FakeRunner(), cfg, log=lambda *_: None, resume=(out, meta))
    recs = load_records(out2 / "runs.jsonl")
    assert out2 == out and [r.task_id for r in recs] == ["qiskitHumanEval/1", "qiskitHumanEval/2"]
    assert len(provider.calls) == 1 and m["modes"]["baseline"]["n_runs"] == 2
    meta2 = json.loads((out / "metadata.json").read_text())
    assert "aborted" not in meta2 and len(meta2["resumed_at"]) == 1
    with pytest.raises(ValueError, match="task versions changed"):   # unit tasks differ from the real dataset
        load_resume(out)


def test_resume_refuses_changed_sandbox(cfg):
    tasks = [make_task(task_id="qiskitHumanEval/1")]
    good = "```python\n# GOOD\nx=1\n```"
    out, _ = evaluate(tasks, ["baseline"], ScriptedProvider([good]), FakeRunner(), cfg, log=lambda *_: None)
    meta = json.loads((out / "metadata.json").read_text())

    class NewImage(FakeRunner):
        def check_available(self):
            return {"image_id": "different"}
    with pytest.raises(ValueError, match="image_id"):
        evaluate(tasks, ["baseline"], ScriptedProvider([]), NewImage(), cfg, log=lambda *_: None, resume=(out, meta))


def test_load_resume_restores_original_config(cfg, tmp_path):
    from qiskit_repair_agent.evaluator import build_metadata, load_resume
    from qiskit_repair_agent.tasks import select_tasks
    c = Config(provider="openrouter", model="x/y:free", max_repairs=1, results_dir=str(tmp_path))
    tasks = select_tasks("standard", limit=2)
    meta = build_metadata("e1", c, tasks, ["baseline", "repair"], 3, ScriptedProvider([]), {"image_id": "i"})
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    cfg2, tasks2, modes, repeats, _ = load_resume(tmp_path)
    assert cfg2 == c and [t.task_id for t in tasks2] == [t.task_id for t in tasks]
    assert (modes, repeats) == (["baseline", "repair"], 3)


def test_pass_rate_by_repair_budget(records):
    r = compute_metrics(records)["modes"]["repair"]
    # repair runs: task1 pass@0, task2 pass@1, task3 fail after 2, task4 provider error -> n = 4
    assert r["pass_rate_by_repair_budget"] == {"0": 0.25, "1": 0.5, "2": 0.5}
