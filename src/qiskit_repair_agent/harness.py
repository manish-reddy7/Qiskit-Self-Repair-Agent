"""Test harness. Runs INSIDE the sandbox container (stdlib only; source is passed via `python -I -c`).

Reads a JSON payload {prelude, code, test, entry_point, max_chars} from stdin, then in one namespace:
prelude -> generated code -> hidden test -> check(entry_point). Prints exactly one result line
prefixed with MARKER to the original stdout. All program output (fd 1 and 2) is redirected to a
temp file in the size-limited tmpfs and only the first max_chars are reported.

Limitation: generated code shares this interpreter, so deliberately hostile code could forge a
result. The sandbox contains the damage; the harness only targets honest mistakes.
"""
import json
import linecache
import os
import re
import sys
import tempfile
import traceback

MARKER = "@@QRA_RESULT@@"
QISKIT_MODULES = ("qiskit", "qiskit_aer", "qiskit_ibm_runtime", "qiskit_experiments", "qiskit_ibm_transpiler")


def classify(exc):
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "import_error"
    if isinstance(exc, (MemoryError, RecursionError)):
        return "resource_error"
    if isinstance(exc, AssertionError):
        return "test_failure"
    if type(exc).__module__.split(".")[0] in QISKIT_MODULES:
        return "api_misuse"
    if isinstance(exc, (AttributeError, TypeError, ValueError)):
        frames = traceback.extract_tb(exc.__traceback__)
        if any(m in (f.filename or "") for f in frames for m in QISKIT_MODULES):
            return "api_misuse"
        if isinstance(exc, AttributeError) and any(m in str(exc) for m in ("qiskit", "Circuit", "Gate")):
            return "api_misuse"
    return "runtime_error"


def main():
    payload = json.loads(sys.stdin.read())
    max_chars = int(payload.get("max_chars", 4000))
    result = {"passed": False, "category": None, "stage": None, "error_type": None,
              "message": "", "traceback": "", "qiskit_version": None}

    real_stdout = os.dup(1)
    capture = tempfile.TemporaryFile()
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(capture.fileno(), 1)
    os.dup2(capture.fileno(), 2)

    ns = {"__name__": "__main__"}
    stage = "prelude"
    try:
        if payload["prelude"]:
            exec(compile(payload["prelude"], "<task_prompt>", "exec"), ns)
        stage = "compile"
        code = compile(payload["code"], "solution.py", "exec")
        # Make solution source visible in tracebacks (hidden test source is deliberately not).
        lines = payload["code"].splitlines(True)
        linecache.cache["solution.py"] = (len(payload["code"]), None, lines, "solution.py")
        stage = "load"
        exec(code, ns)
        stage = "entry_point"
        if not callable(ns.get(payload["entry_point"])):
            result.update(category="invalid_output", stage=stage, error_type="MissingEntryPoint",
                          message=f"function {payload['entry_point']!r} is not defined by the solution")
        else:
            stage = "test"
            exec(compile(payload["test"], "<hidden_test>", "exec"), ns)
            if not re.search(r"^check\(", payload["test"], re.M):   # hard-variant tests call check() themselves
                ns["check"](ns[payload["entry_point"]])
            result.update(passed=True, stage="test")
    except SyntaxError as e:
        result.update(category="syntax_error" if stage == "compile" else "environment_error", stage=stage,
                      error_type=type(e).__name__, message=f"{e.msg} (line {e.lineno})",
                      traceback="".join(traceback.format_exception_only(type(e), e)))
    except BaseException as e:   # noqa: BLE001 - includes SystemExit from generated code
        if isinstance(e, KeyboardInterrupt):
            raise
        category = "environment_error" if stage == "prelude" else classify(e)
        result.update(category=category, stage=stage, error_type=type(e).__name__, message=str(e)[:max_chars],
                      traceback="".join(traceback.format_exception(type(e), e, e.__traceback__))[-max_chars:])

    mod = sys.modules.get("qiskit")
    result["qiskit_version"] = getattr(mod, "__version__", None)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    capture.seek(0)
    result["output"] = capture.read(max_chars + 1).decode("utf-8", "replace")
    if len(result["output"]) > max_chars:
        result["output"] = result["output"][:max_chars] + "\n...[output truncated]"
    os.write(real_stdout, ("\n" + MARKER + json.dumps(result) + "\n").encode())
    os._exit(0)   # skip interpreter teardown: generated code may have left threads/atexit hooks


if __name__ == "__main__":
    main()
