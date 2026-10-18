"""Versioned prompt templates. Bump PROMPT_VERSION on ANY wording change: results across versions are not comparable."""
from __future__ import annotations

from .models import ExecResult, Task

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are an expert Qiskit developer writing Python for Qiskit 2.4.

The code runs in a sandbox with Python 3.12, qiskit 2.4.2, qiskit-aer 0.17, qiskit-ibm-runtime 0.45 \
(fake backends only), numpy, scipy, matplotlib and networkx. There is no network access, no IBM \
Quantum account, and no persistent filesystem.

Reply with exactly one ```python fenced code block containing a complete, self-contained module: \
all imports plus the requested function, with exactly the requested name and signature. \
Do not include tests, example usage, or explanations."""


def initial_prompt(task: Task) -> str:
    return (f"Implement the following task. The required function name is `{task.entry_point}`.\n\n"
            f"<task>\n{task.prompt.strip()}\n</task>")


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else "...[earlier lines truncated]\n" + text[-limit:]


def failure_summary(code: str | None, result: ExecResult | None, error: str | None, limit: int) -> str:
    """Concise, structured feedback for the repair prompt. Contains error details only, never test source."""
    if result is None:   # no executable code was extracted from the previous reply
        return f"Category: invalid_output\nProblem: {error or 'no Python code block was found'}"
    lines = [f"Category: {result.category}"]
    if result.category == "timeout":
        lines.append(f"Problem: {result.message}. Avoid unbounded loops and large simulations.")
    else:
        where = {"compile": "while compiling your code", "load": "while importing your module",
                 "entry_point": "after loading your module", "test": "while the hidden tests called your function",
                 "sandbox": "in the sandbox process"}.get(result.stage or "", "")
        lines.append(f"Error {where}: {result.error_type}: {result.message}".replace("  ", " "))
    if result.traceback and result.category != "syntax_error":
        lines.append("Traceback (hidden test frames show no source):\n" + _clip(result.traceback, limit))
    elif result.traceback:
        lines.append(_clip(result.traceback, limit))
    if result.output.strip():
        lines.append("Program output:\n" + _clip(result.output, limit // 2))
    return "\n".join(lines)


def repair_prompt(task: Task, previous_code: str | None, summary: str) -> str:
    previous = f"```python\n{previous_code}```" if previous_code else "(your previous reply contained no usable code)"
    return (f"{initial_prompt(task)}\n\n"
            f"Your previous solution:\n{previous}\n\n"
            f"It failed when executed against the hidden tests:\n<failure>\n{summary}\n</failure>\n\n"
            "Diagnose the failure and reply with the complete corrected module in one ```python block.")
