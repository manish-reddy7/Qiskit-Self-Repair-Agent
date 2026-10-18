"""Task registry for the vendored Qiskit HumanEval benchmark (Apache-2.0, see benchmark/NOTICE.md)."""
from __future__ import annotations

import hashlib
import json
import re
from functools import cache
from pathlib import Path

from .models import Task

BENCHMARK_DIR = Path(__file__).parent / "benchmark"
DATASET_FILES = {"standard": "qiskit_humaneval.json", "hard": "qiskit_humaneval_hard.json"}
REQUIRED_FIELDS = ("task_id", "prompt", "canonical_solution", "test", "entry_point")
TASK_ID_RE = re.compile(r"qiskitHumanEval/\d{1,4}")


class TaskError(ValueError):
    pass


def _make_task(raw: dict, dataset: str) -> Task:
    missing = [f for f in REQUIRED_FIELDS if not isinstance(raw.get(f), str) or not raw[f].strip()]
    if missing:
        raise TaskError(f"task {raw.get('task_id', '?')!r}: missing or empty fields {missing}")
    if not TASK_ID_RE.fullmatch(raw["task_id"]):
        raise TaskError(f"invalid task id {raw['task_id']!r}")
    if not raw["entry_point"].isidentifier():
        raise TaskError(f"task {raw['task_id']}: entry_point {raw['entry_point']!r} is not an identifier")
    if "def check(" not in raw["test"]:
        raise TaskError(f"task {raw['task_id']}: test does not define check(candidate)")
    # Standard prompts are an importable stub (imports + signature + docstring), as in HumanEval:
    # the stub runs first so the test sees the prompt's imports. Hard prompts are prose.
    prelude = raw["prompt"] if dataset == "standard" else ""
    reference = raw["prompt"] + raw["canonical_solution"] if dataset == "standard" else raw["canonical_solution"]
    digest = hashlib.sha256(
        "\0".join([raw["prompt"], prelude, raw["test"], raw["entry_point"]]).encode()
    ).hexdigest()[:12]
    return Task(
        task_id=raw["task_id"], version=digest, prompt=raw["prompt"], prelude=prelude, test=raw["test"],
        entry_point=raw["entry_point"], reference=reference,
        difficulty=raw.get("difficulty_scale", "unknown"), source=f"qiskit_humaneval_{dataset}",
    )


def load_tasks_from_file(path: Path, dataset: str) -> list[Task]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise TaskError(f"{path}: expected a non-empty JSON list of tasks")
    tasks = [_make_task(raw, dataset) for raw in data]
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise TaskError(f"{path}: duplicate task ids")
    return tasks


@cache
def load_tasks(dataset: str = "standard") -> tuple[Task, ...]:
    if dataset not in DATASET_FILES:
        raise TaskError(f"unknown dataset {dataset!r}; choose from {list(DATASET_FILES)}")
    return tuple(load_tasks_from_file(BENCHMARK_DIR / DATASET_FILES[dataset], dataset))


def dataset_sha256(dataset: str) -> str:
    return hashlib.sha256((BENCHMARK_DIR / DATASET_FILES[dataset]).read_bytes()).hexdigest()


def normalize_task_id(value: str) -> str:
    """Accept 'qiskitHumanEval/7' or just '7'."""
    value = value.strip()
    if value.isdigit():
        value = f"qiskitHumanEval/{int(value)}"
    if not TASK_ID_RE.fullmatch(value):
        raise TaskError(f"invalid task id {value!r} (expected e.g. qiskitHumanEval/7 or 7)")
    return value


def get_task(task_id: str, dataset: str = "standard") -> Task:
    task_id = normalize_task_id(task_id)
    for t in load_tasks(dataset):
        if t.task_id == task_id:
            return t
    raise TaskError(f"task {task_id} not found in dataset {dataset!r}")


# ---- validation manifest (written by `validate-tasks`, committed alongside the dataset) ----

def validation_path(dataset: str) -> Path:
    return BENCHMARK_DIR / f"validation_{dataset}.json"


def load_validation(dataset: str) -> dict:
    p = validation_path(dataset)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def is_validated(task: Task, validation: dict) -> bool:
    """Usable = reference passes, a null stub fails, and the task has not changed since validation."""
    v = validation.get("tasks", {}).get(task.task_id)
    return bool(v) and v["version"] == task.version and v["reference_passed"] and not v["stub_passed"]


def select_tasks(dataset: str, ids: list[str] | None = None, difficulty: str | None = None,
                 limit: int | None = None, include_unvalidated: bool = False) -> list[Task]:
    tasks = [get_task(i, dataset) for i in ids] if ids else list(load_tasks(dataset))
    if difficulty:
        tasks = [t for t in tasks if t.difficulty == difficulty]
    if not include_unvalidated:
        validation = load_validation(dataset)
        if not validation:
            raise TaskError(f"no validation manifest for dataset {dataset!r}; run `validate-tasks` first "
                            "or pass --include-unvalidated")
        bad = [t.task_id for t in tasks if not is_validated(t, validation)]
        if ids and bad:
            raise TaskError(f"tasks not validated (reference fails, test is vacuous, or task changed): {bad}")
        tasks = [t for t in tasks if is_validated(t, validation)]
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise TaskError("task selection is empty")
    return tasks
