"""Result persistence (JSONL), metrics, and JSON/CSV/Markdown exports."""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Attempt, ExecResult, RunRecord

EXCLUDED_STATUSES = ("provider_error", "blocked", "internal_error")   # no valid solution was evaluated


# ---- persistence ----

def append_record(path: Path, record: RunRecord) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(record)) + "\n")
        f.flush()


def load_records(path: Path) -> list[RunRecord]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d["attempts"] = [Attempt(**{**a, "execution": ExecResult(**a["execution"]) if a["execution"] else None})
                         for a in d["attempts"]]
        records.append(RunRecord(**d))
    return records


# ---- metrics ----

def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _known_sum(values) -> int | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def _mode_metrics(rs: list[RunRecord]) -> dict:
    n = len(rs)
    valid = [r for r in rs if r.final_status not in EXCLUDED_STATUSES]
    first_fail = [r for r in valid if not r.initial_passed]
    latencies = [r.total_latency_s for r in rs]
    costs = [r.cost_usd for r in rs]
    return {
        "n_runs": n,
        "n_valid": len(valid),
        "status_counts": dict(Counter(r.final_status for r in rs)),
        # all-runs denominator: provider/sandbox failures count as not passed (nothing is silently dropped)
        "first_attempt_pass_rate": _rate(sum(r.initial_passed for r in rs), n),
        "final_pass_rate": _rate(sum(r.final_passed for r in rs), n),
        # valid-runs denominator: excludes runs where no solution could be evaluated
        "first_attempt_pass_rate_valid": _rate(sum(r.initial_passed for r in valid), len(valid)),
        "final_pass_rate_valid": _rate(sum(r.final_passed for r in valid), len(valid)),
        "n_initially_failing": len(first_fail),
        "repair_success_rate": _rate(sum(r.final_passed for r in first_fail), len(first_fail)),
        "mean_repairs": statistics.fmean([r.repairs_used for r in rs]) if rs else None,
        "mean_repairs_initially_failing":
            statistics.fmean([r.repairs_used for r in first_fail]) if first_fail else None,
        "repairs_distribution": {str(k): v for k, v in sorted(Counter(r.repairs_used for r in rs).items())},
        "latency_mean_s": statistics.fmean(latencies) if rs else None,
        "latency_median_s": statistics.median(latencies) if rs else None,
        # totals over runs that reported usage; None when no run did (unknown, not zero)
        "input_tokens_total": _known_sum(r.input_tokens for r in rs),
        "output_tokens_total": _known_sum(r.output_tokens for r in rs),
        "runs_with_unknown_tokens": sum(r.input_tokens is None for r in rs),
        "cost_usd_total": None if any(c is None for c in costs) or not rs else sum(costs),   # None = unknown
        "failure_distribution_all_attempts": dict(Counter(a.category for r in rs for a in r.attempts if not a.passed)),
        "failure_distribution_initial": dict(Counter(r.attempts[0].category for r in rs
                                                     if r.attempts and not r.attempts[0].passed)),
        "failure_distribution_final": dict(Counter((r.attempts[-1].category if r.attempts else r.final_status)
                                                   for r in rs if not r.final_passed)),
    }


def mcnemar_exact_p(b: int, c: int) -> float | None:
    """Two-sided exact McNemar test (binomial on the discordant pairs b, c under p=0.5)."""
    n = b + c
    if n == 0:
        return None
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def compute_metrics(records: list[RunRecord]) -> dict:
    by_mode = {m: [r for r in records if r.mode == m] for m in sorted({r.mode for r in records})}
    out: dict[str, Any] = {"modes": {m: _mode_metrics(rs) for m, rs in by_mode.items()}}
    if "baseline" in by_mode and "repair" in by_mode:
        b, s = out["modes"]["baseline"], out["modes"]["repair"]
        pp = lambda x, y: None if x is None or y is None else round(100 * (x - y), 2)  # noqa: E731
        base = {(r.task_id, r.repeat): r for r in by_mode["baseline"]}
        pairs = [(base[(r.task_id, r.repeat)], r) for r in by_mode["repair"] if (r.task_id, r.repeat) in base]
        repair_only = sum(not x.final_passed and y.final_passed for x, y in pairs)
        baseline_only = sum(x.final_passed and not y.final_passed for x, y in pairs)
        repeats = sorted({r.repeat for _, r in pairs})
        by_repeat = []
        for rep in repeats:
            rp = [(x, y) for x, y in pairs if y.repeat == rep]
            by_repeat.append(round(100 * (sum(y.final_passed for _, y in rp) - sum(x.final_passed for x, _ in rp))
                                   / len(rp), 2))
        out["comparison"] = {
            "mcnemar_exact_p": mcnemar_exact_p(repair_only, baseline_only),
            # pairs from different repeats of the same task are not independent; with repeats > 1 the
            # p-value is optimistic. Prefer the per-repeat spread below, or test a single repeat.
            "mcnemar_pairs_independent": len(repeats) <= 1,
            "improvement_pp_by_repeat": by_repeat,
            "improvement_pp_stdev": round(statistics.stdev(by_repeat), 2) if len(by_repeat) > 1 else None,
            "absolute_improvement_pp": pp(s["final_pass_rate"], b["final_pass_rate"]),
            "absolute_improvement_valid_pp": pp(s["final_pass_rate_valid"], b["final_pass_rate_valid"]),
            "initial_pass_rate_difference_pp": pp(s["first_attempt_pass_rate"], b["first_attempt_pass_rate"]),
            "n_pairs": len(pairs),
            # paired outcome table (same task, same repeat); discordant cells are what a McNemar test would use
            "paired": {
                "both_pass": sum(x.final_passed and y.final_passed for x, y in pairs),
                "repair_only": sum(not x.final_passed and y.final_passed for x, y in pairs),
                "baseline_only": sum(x.final_passed and not y.final_passed for x, y in pairs),
                "neither": sum(not x.final_passed and not y.final_passed for x, y in pairs),
            },
            "latency_ratio_mean": (s["latency_mean_s"] / b["latency_mean_s"]) if b["latency_mean_s"] else None,
            "output_tokens_ratio": (s["output_tokens_total"] / b["output_tokens_total"])
            if b["output_tokens_total"] and s["output_tokens_total"] is not None else None,
        }
    return out


# ---- exports ----

RUN_COLUMNS = ("run_id", "eval_id", "task_id", "task_version", "dataset", "mode", "repeat", "model_id", "provider",
               "prompt_version", "timestamp", "final_status", "stop_reason", "initial_passed", "final_passed",
               "repairs_used", "total_latency_s", "input_tokens", "output_tokens", "cost_usd", "error")
ATTEMPT_COLUMNS = ("run_id", "task_id", "mode", "repeat", "index", "kind", "retry_reason", "passed", "category",
                   "stage", "error_type", "message", "code_sha256", "llm_model", "llm_request_id", "input_tokens",
                   "output_tokens", "cost_usd", "llm_latency_s", "exec_duration_s", "duration_s", "llm_error")


def write_exports(records: list[RunRecord], out_dir: Path, metadata: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    metrics = compute_metrics(records)
    summary = {"metadata": metadata or {}, "metrics": metrics,
               "per_task": [{c: getattr(r, c) for c in RUN_COLUMNS} for r in records]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out_dir / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        w.writeheader()
        for r in records:
            w.writerow({c: getattr(r, c) for c in RUN_COLUMNS})
    with open(out_dir / "attempts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ATTEMPT_COLUMNS)
        w.writeheader()
        for r in records:
            for a in r.attempts:
                e = a.execution
                w.writerow({"run_id": r.run_id, "task_id": r.task_id, "mode": r.mode, "repeat": r.repeat,
                            "index": a.index, "kind": a.kind, "retry_reason": a.retry_reason, "passed": a.passed,
                            "category": a.category, "stage": e and e.stage, "error_type": e and e.error_type,
                            "message": (e.message[:300] if e else None), "code_sha256": a.code_sha256,
                            "llm_model": a.llm_model, "llm_request_id": a.llm_request_id,
                            "input_tokens": a.input_tokens, "output_tokens": a.output_tokens,
                            "cost_usd": a.cost_usd, "llm_latency_s": round(a.llm_latency_s, 3),
                            "exec_duration_s": e and round(e.duration_s, 3), "duration_s": round(a.duration_s, 3),
                            "llm_error": a.llm_error})
    (out_dir / "report.md").write_text(render_markdown(metrics, metadata or {}, records), encoding="utf-8")
    return metrics


def _fmt(v, pct=False):
    if v is None:
        return "n/a"
    if pct:
        return f"{100 * v:.1f}%"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def render_markdown(metrics: dict, meta: dict, records: list[RunRecord]) -> str:
    modes = metrics["modes"]
    names = list(modes)
    rows = [
        ("Runs (tasks x repeats)", "n_runs", False), ("Valid runs (a solution was evaluated)", "n_valid", False),
        ("First-attempt pass rate", "first_attempt_pass_rate", True), ("Final pass rate", "final_pass_rate", True),
        ("Final pass rate (valid runs)", "final_pass_rate_valid", True),
        ("Repair success rate", "repair_success_rate", True), ("Mean repair attempts", "mean_repairs", False),
        ("Latency mean (s)", "latency_mean_s", False), ("Latency median (s)", "latency_median_s", False),
        ("Input tokens (total)", "input_tokens_total", False), ("Output tokens (total)", "output_tokens_total", False),
        ("Provider-reported cost (USD)", "cost_usd_total", False),
    ]
    lines = ["# Evaluation report", ""]
    if meta.get("provider") == "mock":
        lines += ["> **MOCK PROVIDER: pipeline check only, not an LLM result.** The mock returns a failing stub "
                  "first and the reference solution on repair, so these numbers are scripted.", ""]
    for k in ("eval_id", "date", "model", "provider", "dataset", "prompt_version", "max_repairs", "repeats",
              "n_tasks", "qiskit_version", "image_id"):
        if k in meta:
            lines.append(f"- **{k}**: {meta[k]}")
    lines += ["", "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, key, pct in rows:
        lines.append(f"| {label} | " + " | ".join(_fmt(modes[m][key], pct) for m in names) + " |")
    if "comparison" in metrics:
        c = metrics["comparison"]
        lines += ["", "## Baseline vs self-repair (paired by task and repeat)", "",
                  f"- Absolute improvement in final pass rate: **{_fmt(c['absolute_improvement_pp'])} pp** "
                  f"(valid runs only: {_fmt(c['absolute_improvement_valid_pp'])} pp)",
                  f"- Difference in initial pass rate (both modes generate independently): "
                  f"{_fmt(c['initial_pass_rate_difference_pp'])} pp",
                  f"- Paired outcomes over {c['n_pairs']} pairs: {c['paired']}",
                  f"- Exact McNemar test on discordant pairs: p = {_fmt(c['mcnemar_exact_p'])}"
                  + ("" if c["mcnemar_pairs_independent"] else
                     " (pairs from repeats of the same task are not independent: this p-value is optimistic)"),
                  f"- Improvement by repeat (pp): {c['improvement_pp_by_repeat']}; "
                  f"stdev: {_fmt(c['improvement_pp_stdev'])}",
                  f"- Latency ratio (repair/baseline, mean): {_fmt(c['latency_ratio_mean'])}; "
                  f"output-token ratio: {_fmt(c['output_tokens_ratio'])}"]
    lines += ["", "## Failure categories", ""]
    for m in names:
        lines.append(f"- **{m}** initial: {modes[m]['failure_distribution_initial']}; "
                     f"final: {modes[m]['failure_distribution_final']}")
    lines += ["", "## Per-task outcomes", "", "| Task | Repeat | Mode | Status | Repairs | Latency (s) |",
              "|---|---|---|---|---|---|"]
    for r in sorted(records, key=lambda r: (int(r.task_id.split("/")[1]), r.repeat, r.mode)):
        lines.append(f"| {r.task_id} | {r.repeat} | {r.mode} | {r.final_status} | {r.repairs_used} | "
                     f"{r.total_latency_s:.1f} |")
    lines += ["", "## Limitations", "",
              "- Single model; results are specific to the model, prompt version and settings listed above.",
              "- Unless repeats > 1, every number comes from a single stochastic run. The McNemar p-value "
              "tests only the paired task outcomes of this run; it does not account for model nondeterminism.",
              "- Repair feedback includes hidden-test assertion messages, so repair mode receives information "
              "the baseline does not; this is the intended experimental difference.",
              "- Tasks that need IBM Quantum network access or whose reference fails in the sandbox are excluded "
              "by validation (see the validation manifest).", ""]
    return "\n".join(lines)
