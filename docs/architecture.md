# Architecture

```
task (registry) ─► prompts.initial_prompt ─► Provider.generate ─► llm.extract_code ─► DockerRunner.execute
                                                    ▲                                          │
                                                    │  repair mode only, ≤ max_repairs         ▼
                    prompts.repair_prompt ◄── prompts.failure_summary ◄──────────── ExecResult (harness)
```

| Module | Responsibility |
|---|---|
| `config.py` | Frozen, validated `Config`. Defaults < TOML `[agent]` < `QRA_*` env < CLI. Holds no secrets |
| `tasks.py` | Loads the vendored Qiskit HumanEval JSON, validates its schema, computes content-hash task versions, normalises ids, and selects tasks using the validation manifest |
| `prompts.py` | Versioned (`PROMPT_VERSION`) system, initial and repair prompts, plus the structured failure summary |
| `llm.py` | `Provider` protocol, `AnthropicProvider`, `OpenRouterProvider` (stdlib HTTP, free-tier pacing), `ScriptedProvider` (tests), `ReferenceMockProvider` (offline demo), `extract_code` |
| `controller.py` | `run_task`: baseline and repair loops, stop conditions, per-attempt trace, final status |
| `runner.py` | Builds the hardened `docker run` command, sends the payload on stdin, enforces the timeout (`docker kill`), parses the result |
| `harness.py` | Runs **inside** the container: prelude → solution → hidden test → `check()`. Classifies failures and bounds output |
| `evaluator.py` | `evaluate` (modes interleaved per task, incremental JSONL, survives task crashes, stops on fatal provider errors, `--resume`) and `validate_tasks` |
| `reporting.py` | JSONL persistence, metrics (incl. exact McNemar), and JSON/CSV/Markdown export |
| `dashboard/app.py` | Optional read-only Streamlit browser over `results/` |
| `cli.py` | argparse commands: `list-tasks`, `validate-tasks`, `run`, `evaluate`, `report`, `clean` |

The PRD's suggested `llm/` sub-package, `result_store.py` and `scripts/` were folded into `llm.py`,
`reporting.py` and the CLI. They would otherwise be files of a few lines each.

## Execution sandbox

`docker run --rm -i --network none --user 10001:10001 --read-only --tmpfs /tmp:size=64m
--memory 1g --memory-swap 1g --cpus 1 --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges
--ipc private --shm-size 64m qiskit-repair-runner:2.4.2 python -I -c <harness>`

- No volumes, no `--env`, no Docker socket. The payload (prelude, code, hidden test) arrives on stdin.
  Host environment variables, including `ANTHROPIC_API_KEY`, never enter the container. A test asserts this.
- The harness redirects fds 1/2 to a tmpfs file and reports only the first `max_output_chars`.
- On timeout, the host kills the container by name. `--rm` removes it.
- Exit code 125 (Docker could not start the container) → `blocked`: the code never ran. Exit 137
  without a result → `resource_error` (OOM kill).
- If Docker is missing or the daemon is down, `RunnerUnavailable` is raised. There is no host fallback.
  `evaluate` and `run` check this before making any LLM call.
- `/dev/shm` is kept, capped at 64 MB, because Qiskit's `parallel_map` needs POSIX semaphores. With
  `--ipc none`, task 32's canonical solution failed. That is the kind of sandbox artifact that
  validation exists to catch.

Residual risks: shared kernel (container escape), CPU side channels, and generated code sharing the
harness interpreter (it could forge a pass). Mitigations if this ever faces hostile input: gVisor/Kata
runtime, a separate grader process, seccomp profile review.

## Failure classification (`harness.classify`)

| Category | Rule |
|---|---|
| `syntax_error` | `compile()` of the solution fails |
| `import_error` | `ImportError` / `ModuleNotFoundError` |
| `api_misuse` | Exception class defined in a Qiskit package, or `AttributeError`/`TypeError`/`ValueError` raised from a Qiskit frame, or an `AttributeError` naming Qiskit objects |
| `test_failure` | `AssertionError` (hidden test assertion) |
| `resource_error` | `MemoryError`, `RecursionError`, or the process was killed (137) |
| `timeout` | Exceeded `exec_timeout_s` |
| `runtime_error` | Any other exception, including `SystemExit` from the solution |
| `invalid_output` | No code in the reply (the attempt is not executed), or the entry-point function is missing |
| `environment_error` | The benchmark's own prelude failed, or the container could not run the harness |
| `provider_error` / `blocked` | No code was produced / no code was executed |

The heuristic is imperfect. For example, an `AssertionError` raised by the solution itself counts as
`test_failure`.

## Stop conditions (repair mode)

In order of evaluation after each attempt: passed → provider error → blocked → (baseline: always stop)
→ `max_repairs` reached → per-task time budget → per-task token budget → otherwise repair.
Fatal provider errors (authentication, unknown model, bad request) raise and end the evaluation, which
keeps its partial results. Transient errors are retried by the SDK (`provider_max_retries`, exponential
backoff) and then recorded as `provider_error`.
