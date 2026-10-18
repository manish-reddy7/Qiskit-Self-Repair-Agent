"""Plain data records shared across the package. Everything here is JSON-serialisable via asdict()."""
from __future__ import annotations

from dataclasses import dataclass, field

# Failure categories (PRD FR-11). "passed" is not a category; a passing attempt has category None.
CATEGORIES = (
    "syntax_error",       # generated code does not compile
    "import_error",       # ImportError / ModuleNotFoundError in generated code
    "api_misuse",         # exception raised by or inside Qiskit (wrong/removed API, bad arguments)
    "runtime_error",      # any other exception in the generated code or while the test calls it
    "test_failure",       # code ran, hidden test assertion failed
    "timeout",            # killed after the per-attempt time limit
    "resource_error",     # memory/recursion/process limits, or killed by the container runtime
    "invalid_output",     # model reply had no usable code, or the required function is missing
    "environment_error",  # benchmark prelude/test could not run (sandbox problem, not the model's fault)
    "provider_error",     # LLM API failure or refusal; no code was produced
    "blocked",            # sandbox unavailable; code was NOT executed
)

FINAL_STATUSES = ("passed", "failed", "timed_out", "blocked", "provider_error", "internal_error")


@dataclass(frozen=True)
class Task:
    task_id: str
    version: str          # content hash of prompt/prelude/test/entry_point; changes if the task changes
    prompt: str           # shown to the model
    prelude: str          # executed before the solution (standard dataset: the stub prompt with its imports)
    test: str             # hidden; never shown to the model
    entry_point: str
    reference: str        # full reference module, used only by validate-tasks and the offline mock
    difficulty: str
    source: str           # dataset name


@dataclass
class LLMResponse:
    text: str
    model: str
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None   # None = provider did not report it; never estimated
    stop_reason: str | None = None


class ProviderError(Exception):
    """LLM call failed. fatal=True means retrying other tasks is pointless (auth, bad model, bad request)."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


@dataclass
class ExecResult:
    passed: bool
    category: str | None
    stage: str | None = None          # prelude | compile | load | entry_point | test | sandbox
    error_type: str | None = None
    message: str = ""
    traceback: str = ""
    output: str = ""                  # combined stdout/stderr of the program, truncated
    exit_code: int | None = None
    timed_out: bool = False
    duration_s: float = 0.0
    qiskit_version: str | None = None


@dataclass
class Attempt:
    index: int                        # 0 = initial generation, 1.. = repairs
    kind: str                         # initial | repair
    retry_reason: str | None          # category of the previous attempt that triggered this repair
    passed: bool
    category: str | None
    code: str | None
    code_sha256: str | None
    llm_model: str | None = None
    llm_request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    llm_latency_s: float = 0.0
    llm_error: str | None = None
    execution: ExecResult | None = None
    duration_s: float = 0.0


@dataclass
class RunRecord:
    run_id: str
    eval_id: str
    task_id: str
    task_version: str
    dataset: str
    mode: str                         # baseline | repair
    repeat: int
    model_id: str
    provider: str
    prompt_version: str
    timestamp: str
    final_status: str
    stop_reason: str
    initial_passed: bool
    final_passed: bool
    repairs_used: int
    total_latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    attempts: list[Attempt] = field(default_factory=list)
    error: str | None = None          # only for internal_error
