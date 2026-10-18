"""Benchmark evaluation (both modes, same tasks) and reference validation of the task suite."""
from __future__ import annotations

import json
import platform
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
from importlib import metadata as importlib_metadata
from pathlib import Path

from . import __version__
from .config import Config
from .controller import run_task
from .models import ProviderError, RunRecord, Task
from .prompts import PROMPT_VERSION
from .reporting import append_record, load_records, write_exports
from .tasks import dataset_sha256, get_task, load_validation, validation_path


def _pkg_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def new_eval_dir(cfg: Config, label: str = "eval") -> tuple[str, Path]:
    eval_id = f"{label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = Path(cfg.results_dir) / eval_id
    out.mkdir(parents=True, exist_ok=False)
    return eval_id, out


def build_metadata(eval_id: str, cfg: Config, tasks: list[Task], modes: list[str], repeats: int,
                   provider, runner_info: dict) -> dict:
    return {
        "eval_id": eval_id, "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": provider.model, "provider": provider.name, "dataset": cfg.dataset,
        "dataset_sha256": dataset_sha256(cfg.dataset), "prompt_version": PROMPT_VERSION,
        "max_repairs": cfg.max_repairs, "repeats": repeats, "modes": modes, "n_tasks": len(tasks),
        "tasks": {t.task_id: t.version for t in tasks}, "config": cfg.public_dict(),
        "versions": {"qiskit_repair_agent": __version__, "python": sys.version.split()[0],
                     "platform": platform.platform(), "anthropic_sdk": _pkg_version("anthropic"), **runner_info},
        "qiskit_version": "2.4.2 (sandbox image; per-attempt value recorded in each execution)",
        "image_id": runner_info.get("image_id"),
    }


def _internal_error_record(task: Task, mode: str, repeat: int, eval_id: str, provider, exc: Exception) -> RunRecord:
    return RunRecord(run_id=uuid.uuid4().hex, eval_id=eval_id, task_id=task.task_id, task_version=task.version,
                     dataset=task.source, mode=mode, repeat=repeat, model_id=provider.model, provider=provider.name,
                     prompt_version=PROMPT_VERSION, timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
                     final_status="internal_error", stop_reason="internal_error", initial_passed=False,
                     final_passed=False, repairs_used=0, total_latency_s=0.0, input_tokens=None,
                     output_tokens=None, cost_usd=None,
                     error="".join(traceback.format_exception_only(type(exc), exc)).strip())


def load_resume(out: Path) -> tuple[Config, list[Task], list[str], int, dict]:
    """Settings for resuming an evaluation come from its own metadata, never from the current CLI,
    so both halves of a resumed run are generated under identical conditions."""
    meta_path = Path(out) / "metadata.json"
    if not meta_path.is_file():
        raise ValueError(f"{out} is not an evaluation directory (no metadata.json)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = Config(**meta["config"])
    if meta["prompt_version"] != PROMPT_VERSION:
        raise ValueError(f"prompt version changed ({meta['prompt_version']} -> {PROMPT_VERSION}); cannot resume")
    if meta["dataset_sha256"] != dataset_sha256(cfg.dataset):
        raise ValueError("benchmark dataset file changed since this evaluation started; cannot resume")
    tasks = [get_task(task_id, cfg.dataset) for task_id in meta["tasks"]]
    changed = [t.task_id for t in tasks if t.version != meta["tasks"][t.task_id]]
    if changed:
        raise ValueError(f"task versions changed since this evaluation started: {changed}")
    return cfg, tasks, meta["modes"], meta["repeats"], meta


def evaluate(tasks: list[Task], modes: list[str], provider, runner, cfg: Config, repeats: int = 1,
             log=print, resume: tuple[Path, dict] | None = None) -> tuple[Path, dict]:
    """Interleaves modes per task so drift (API load, time of day) hits both modes equally.
    Each run is appended to runs.jsonl as soon as it finishes, so partial runs remain analysable.
    resume=(dir, metadata) continues an earlier evaluation, skipping (task, mode, repeat) already recorded."""
    runner_info = runner.check_available()   # refuse to start without a sandbox; never fall back to the host
    if resume:
        out, meta = resume
        for key in ("image_id", "harness_sha256"):
            if meta["versions"].get(key) != runner_info.get(key):
                raise ValueError(f"sandbox {key} changed since this evaluation started; cannot resume")
        meta.pop("aborted", None)
        meta.setdefault("resumed_at", []).append(datetime.now(UTC).isoformat(timespec="seconds"))
    else:
        eval_id, out = new_eval_dir(cfg)
        meta = build_metadata(eval_id, cfg, tasks, modes, repeats, provider, runner_info)
    eval_id = meta["eval_id"]
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    runs_path = out / "runs.jsonl"
    finished = {(r.task_id, r.mode, r.repeat) for r in load_records(runs_path)} if runs_path.exists() else set()
    total, done = len(tasks) * len(modes) * repeats, len(finished)
    if finished:
        log(f"Resuming {eval_id}: {done}/{total} runs already recorded")
    try:
        for repeat in range(repeats):
            for task in tasks:
                for mode in modes:
                    if (task.task_id, mode, repeat) in finished:
                        continue
                    try:
                        rec = run_task(task, mode, provider, runner, cfg, eval_id=eval_id, repeat=repeat)
                    except ProviderError:
                        raise
                    except Exception as e:   # noqa: BLE001 - one broken task must not end the benchmark
                        rec = _internal_error_record(task, mode, repeat, eval_id, provider, e)
                    append_record(runs_path, rec)
                    done += 1
                    log(f"[{done}/{total}] {task.task_id} {mode:8s} rep={repeat} -> {rec.final_status:14s} "
                        f"repairs={rec.repairs_used} {rec.total_latency_s:.1f}s")
    except ProviderError as e:
        log(f"Stopping evaluation: fatal provider error: {e}. Partial results kept in {out}; "
            f"continue later with: evaluate --resume {out}")
        meta["aborted"] = str(e)
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    records = load_records(runs_path) if runs_path.exists() else []
    metrics = write_exports(records, out, meta)
    return out, metrics


def validate_tasks(tasks: list[Task], runner, cfg: Config, workers: int = 4, log=print) -> dict:
    """Runs each reference solution and a null stub in the sandbox. A task is usable only if the
    reference passes and the stub fails (otherwise the test cannot distinguish right from wrong)."""
    runner_info = runner.check_available()

    def one(task: Task) -> tuple[str, dict]:
        ref = runner.execute(task, task.reference)
        stub = runner.execute(task, f"def {task.entry_point}(*args, **kwargs):\n    return None\n")
        return task.task_id, {"version": task.version, "difficulty": task.difficulty,
                              "reference_passed": ref.passed, "reference_category": ref.category,
                              "reference_error": f"{ref.error_type}: {ref.message}"[:300] if not ref.passed else None,
                              "stub_passed": stub.passed}

    previous = load_validation(cfg.dataset)
    results = previous.get("tasks", {}) if previous.get("dataset_sha256") == dataset_sha256(cfg.dataset) else {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task_id, res in pool.map(one, tasks):
            results[task_id] = res
            ok = res["reference_passed"] and not res["stub_passed"]
            log(f"{task_id:22s} {'OK ' if ok else 'BAD'} reference={res['reference_category'] or 'passed'} "
                f"stub_passed={res['stub_passed']}")
    manifest = {"dataset": cfg.dataset, "dataset_sha256": dataset_sha256(cfg.dataset),
                "date": datetime.now(UTC).isoformat(timespec="seconds"),
                "runner": runner_info, "exec_timeout_s": cfg.exec_timeout_s, "memory": cfg.memory,
                "n_tasks": len(results),
                "n_usable": sum(r["reference_passed"] and not r["stub_passed"] for r in results.values()),
                "tasks": dict(sorted(results.items(), key=lambda kv: int(kv[0].split("/")[1])))}
    validation_path(cfg.dataset).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
