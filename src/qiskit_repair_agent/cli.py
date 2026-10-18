"""Command-line interface: python -m qiskit_repair_agent.cli <command> --help"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .config import load_config, load_dotenv
from .controller import run_task
from .evaluator import evaluate, load_resume, new_eval_dir, validate_tasks
from .llm import make_provider
from .models import ProviderError
from .reporting import append_record, load_records, write_exports
from .runner import DockerRunner, RunnerUnavailable
from .tasks import TaskError, get_task, load_tasks, load_validation, is_validated, select_tasks


def _config(args):
    return load_config(args.config, provider=getattr(args, "provider", None), model=getattr(args, "model", None),
                       max_repairs=getattr(args, "max_repairs", None), dataset=getattr(args, "dataset", None))


def cmd_list_tasks(args):
    cfg = _config(args)
    validation = load_validation(cfg.dataset)
    tasks = load_tasks(cfg.dataset)
    usable = 0
    for t in tasks:
        ok = is_validated(t, validation)
        usable += ok
        if ok or args.all:
            first = next((ln for ln in t.prompt.splitlines() if ln.strip() and not ln.startswith(("from", "import"))),
                         "")
            mark = "" if ok else "  [not validated]"
            print(f"{t.task_id:22s} {t.difficulty:12s} {t.entry_point:32s} {first.strip()[:60]}{mark}")
    note = "" if validation else " (no validation manifest; run validate-tasks)"
    print(f"\n{len(tasks)} tasks in dataset {cfg.dataset!r}; {usable} validated and usable{note}")


def cmd_validate(args):
    cfg = _config(args)
    tasks = [get_task(i, cfg.dataset) for i in args.tasks] if args.tasks else list(load_tasks(cfg.dataset))
    m = validate_tasks(tasks, DockerRunner(cfg), cfg, workers=args.workers)
    print(f"\n{m['n_usable']}/{m['n_tasks']} tasks usable; manifest written for dataset {cfg.dataset!r}")


def cmd_run(args):
    cfg = _config(args)
    task = get_task(args.task_id, cfg.dataset)
    if not args.include_unvalidated and not is_validated(task, load_validation(cfg.dataset)):
        raise TaskError(f"{task.task_id} is not validated (see list-tasks --all); pass --include-unvalidated")
    runner = DockerRunner(cfg)
    info = runner.check_available()
    provider = make_provider(cfg, [task])
    eval_id, out = new_eval_dir(cfg, label="run")
    print(f"Running {task.task_id} mode={args.mode} model={provider.model} max_repairs={cfg.max_repairs}")
    rec = run_task(task, args.mode, provider, runner, cfg, eval_id=eval_id)
    append_record(out / "runs.jsonl", rec)
    write_exports([rec], out, {"eval_id": eval_id, "model": provider.model, "provider": provider.name,
                               "dataset": cfg.dataset, "config": cfg.public_dict(), "versions": info})
    for a in rec.attempts:
        e = a.execution
        detail = f"{e.error_type}: {e.message[:150]}" if e and not a.passed else (a.llm_error or "")
        print(f"  attempt {a.index} ({a.kind}): {'PASS' if a.passed else 'FAIL'} "
              f"[{a.category or 'ok'}] {a.duration_s:.1f}s tokens={a.input_tokens}/{a.output_tokens} {detail}")
    print(f"\nFinal: {rec.final_status} (stop: {rec.stop_reason}) in {rec.total_latency_s:.1f}s")
    if args.show_code and rec.attempts[-1].code:
        print("\n--- final code ---\n" + rec.attempts[-1].code)
    print(f"Saved to {out}")
    return 0 if rec.final_passed else 1


def cmd_evaluate(args):
    resume = None
    if args.resume:
        # settings, tasks, modes and repeats all come from the evaluation being resumed
        cfg, tasks, modes, repeats, meta = load_resume(Path(args.resume))
        resume = (Path(args.resume), meta)
    else:
        cfg = _config(args)
        tasks = select_tasks(cfg.dataset, ids=args.tasks, difficulty=args.difficulty, limit=args.limit,
                             include_unvalidated=args.include_unvalidated)
        modes = ["baseline", "repair"] if args.mode == "both" else [args.mode]
        repeats = args.repeats
    provider = make_provider(cfg, tasks)
    print(f"Evaluating {len(tasks)} tasks x {len(modes)} mode(s) x {repeats} repeat(s) "
          f"with {provider.name}:{provider.model}")
    out, metrics = evaluate(tasks, modes, provider, DockerRunner(cfg), cfg, repeats=repeats, resume=resume)
    _print_metrics(metrics)
    print(f"\nResults: {out}  (summary.json, runs.csv, attempts.csv, report.md)")


def _latest_eval_dir(results_dir: str) -> Path:
    dirs = sorted((p for p in Path(results_dir).glob("*/runs.jsonl")), key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise TaskError(f"no results found under {results_dir}/")
    return dirs[-1].parent


def cmd_report(args):
    cfg = _config(args)
    out = Path(args.eval_dir) if args.eval_dir else _latest_eval_dir(cfg.results_dir)
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    metrics = write_exports(load_records(out / "runs.jsonl"), out, meta)
    _print_metrics(metrics)
    print(f"\nReport regenerated in {out}")


def _print_metrics(metrics):
    for mode, m in metrics["modes"].items():
        pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
        print(f"\n[{mode}] runs={m['n_runs']} valid={m['n_valid']} first-pass={pct(m['first_attempt_pass_rate'])} "
              f"final-pass={pct(m['final_pass_rate'])} repair-success={pct(m['repair_success_rate'])} "
              f"median-latency={m['latency_median_s'] and round(m['latency_median_s'], 1)}s "
              f"tokens={m['input_tokens_total']}/{m['output_tokens_total']}")
    if "comparison" in metrics:
        c = metrics["comparison"]
        print(f"\nabsolute improvement: {c['absolute_improvement_pp']} pp; paired: {c['paired']}")


def cmd_clean(args):
    cfg = _config(args)
    target = Path(cfg.results_dir).resolve()
    if not target.exists():
        print(f"{target} does not exist; nothing to delete")
        return
    if target == Path.cwd().resolve() or target in Path.cwd().resolve().parents:
        raise TaskError(f"refusing to delete {target}")
    if not args.yes:
        print(f"Would delete {target} (all local run records). Re-run with --yes to confirm.")
        return
    shutil.rmtree(target)
    print(f"Deleted {target}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m qiskit_repair_agent.cli",
                                description="LLM Qiskit code generation with execution-guided self-repair.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="TOML config file (e.g. configs/default.toml)")
    common.add_argument("--dataset", choices=["standard", "hard"], help="Qiskit HumanEval variant")
    llm = argparse.ArgumentParser(add_help=False)
    llm.add_argument("--provider", choices=["anthropic", "openrouter", "ollama", "mock"])
    llm.add_argument("--model")
    llm.add_argument("--max-repairs", type=int, dest="max_repairs")
    llm.add_argument("--include-unvalidated", action="store_true", help="allow tasks not in the validation manifest")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list-tasks", parents=[common], help="list benchmark tasks")
    s.add_argument("--all", action="store_true", help="include tasks that failed validation")
    s.set_defaults(func=cmd_list_tasks)

    s = sub.add_parser("validate-tasks", parents=[common], help="check reference solutions in the sandbox")
    s.add_argument("--tasks", nargs="+", help="task ids (default: all)")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("run", parents=[common, llm], help="run a single task")
    s.add_argument("task_id", help="e.g. qiskitHumanEval/7 or 7")
    s.add_argument("--mode", choices=["baseline", "repair"], default="repair")
    s.add_argument("--show-code", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("evaluate", parents=[common, llm], help="evaluate the benchmark")
    s.add_argument("--mode", choices=["both", "baseline", "repair"], default="both")
    s.add_argument("--tasks", nargs="+", help="explicit task ids (default: all validated tasks)")
    s.add_argument("--limit", type=int, help="first N selected tasks (for development)")
    s.add_argument("--difficulty", choices=["basic", "intermediate", "difficult"])
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--resume", metavar="EVAL_DIR", help="continue an interrupted evaluation with its original settings")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("report", parents=[common], help="recompute metrics/exports for an evaluation")
    s.add_argument("--eval-dir", help="results/<eval_id> (default: most recent)")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("clean", parents=[common], help="delete local results")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_clean)
    return p


def main(argv=None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (ValueError, TaskError, RunnerUnavailable, ProviderError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
