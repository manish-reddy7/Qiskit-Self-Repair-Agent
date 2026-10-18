# qiskit-self-repair-agent

Research prototype for Quantum Computing (25AM3504), Project 7: **does execution-guided LLM self-repair
improve Qiskit code correctness compared with one-shot generation, and at what latency and token cost?**

An LLM (Anthropic Claude by default) writes Qiskit code for
[Qiskit HumanEval](https://github.com/qiskit-community/qiskit-human-eval) tasks. The code runs in a
locked-down Docker container against the task's hidden tests. The project compares two modes:

- **baseline**: generate once, test, stop.
- **repair**: same first generation. On failure, a structured error/test summary and the previous code go
  back to the model, for up to `max_repairs` (default 2) more attempts.

No experimental result is claimed in this repository. Run the evaluation (below) to produce one.

## Status

| Item | State |
|---|---|
| Benchmark | Official Qiskit HumanEval, vendored (Apache-2.0), 151 tasks, standard and hard variants |
| Validated tasks | 143 / 151 in each variant. The 8 excluded tasks need an IBM Quantum account or network access (see `src/qiskit_repair_agent/benchmark/validation_*.json`) |
| Sandbox | Docker: no network, non-root, read-only root, CPU/memory/PID/time limits, no host mounts or env |
| Tests | 114 pytest tests (8 need Docker, 2 need Streamlit); `ruff` and `mypy` clean |
| Real-LLM evaluation | Smoke test only: 1 task with `nvidia/nemotron-3-super-120b-a12b:free` (passed). No benchmark evaluation yet |

## Requirements

- Python ≥ 3.11 (developed with 3.13 on Windows 11)
- Docker Desktop or Docker Engine (required; generated code is **never** executed on the host)
- An Anthropic API key, only for real-model runs. The tests and the mock provider need none.

The host needs only the `anthropic` SDK. Qiskit lives in the sandbox image, pinned to the benchmark's
own versions: `qiskit==2.4.2`, `qiskit-aer==0.17.0`, `qiskit-ibm-runtime==0.45.0`
(see `docker/requirements-runner.txt`).

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,ui]"        # ui = optional Streamlit dashboard
docker build -t qiskit-repair-runner:2.4.2 docker/     # ~1.1 GB image, a few minutes
pytest                                                  # all tests; Docker tests skip if the image is missing
ruff check . && mypy                                    # lint + type check
```

## Configure

```bash
cp .env.example .env        # then put your key in .env (gitignored), or export ANTHROPIC_API_KEY
```

Settings are resolved as defaults < `--config configs/default.toml` < `QRA_<KEY>` environment
variables < CLI flags. Main keys:

| Key | Default | Meaning |
|---|---|---|
| `provider` | `anthropic` | `anthropic`, `openrouter` or `mock` |
| `model` | `claude-opus-5-5` | Model id (recorded per attempt from the API response) |
| `effort` | `medium` | `output_config.effort`. Opus 5.5 rejects `temperature`, so `temperature` is unset by default |
| `max_repairs` | 2 | Extra attempts after the initial generation (repair mode only) |
| `exec_timeout_s` | 30 | Per-attempt sandbox time limit |
| `memory` / `cpus` / `pids_limit` | 1g / 1.0 / 256 | Container limits |
| `max_run_seconds` / `max_run_tokens` | 900 / 200000 | Per-task budget across all attempts |
| `max_output_chars` | 4000 | Cap on captured output, tracebacks and feedback |
| `dataset` | `standard` | `standard` (function stub + docstring) or `hard` (prose-only prompt) |

## Use

```bash
python -m qiskit_repair_agent.cli list-tasks                 # validated tasks (--all shows excluded ones)
python -m qiskit_repair_agent.cli run 7 --mode baseline      # one task, one shot
python -m qiskit_repair_agent.cli run 7 --mode repair --show-code
python -m qiskit_repair_agent.cli evaluate --mode both --limit 10        # development subset
python -m qiskit_repair_agent.cli evaluate --mode both                   # full validated benchmark (143 tasks)
python -m qiskit_repair_agent.cli evaluate --mode both --repeats 3 --dataset hard
python -m qiskit_repair_agent.cli evaluate --resume results/<eval_id>   # continue an interrupted evaluation
python -m qiskit_repair_agent.cli report                     # regenerate exports for the latest evaluation
python -m qiskit_repair_agent.cli validate-tasks             # re-check reference solutions in the sandbox
python -m qiskit_repair_agent.cli clean --yes                # delete all local results
```

Other options: `--tasks 0 5 12`, `--difficulty basic`, `--model ...`, `--max-repairs N`,
`--config configs/default.toml`.

**Offline demo (no API key):** add `--provider mock` to `run` or `evaluate`. The mock returns a failing
stub first and the reference solution on repair. That exercises prompts, the real sandbox, the repair
loop and the reports, but the numbers it produces are scripted and say nothing about any LLM.
Reports from the mock are labelled accordingly.

**Free models via OpenRouter:** get a key at https://openrouter.ai/keys, set `OPENROUTER_API_KEY`, and pick a
`:free` model. The list changes often; see `https://openrouter.ai/api/v1/models` for ids ending in `:free`.

```bash
python -m qiskit_repair_agent.cli run 7 --mode repair --provider openrouter --model nvidia/nemotron-3-super-120b-a12b:free
python -m qiskit_repair_agent.cli evaluate --mode both --limit 10 --provider openrouter --model nvidia/nemotron-3-super-120b-a12b:free
```

- **Limits:** 20 requests/min (requests are paced automatically) and **50 requests/day**, or 1,000/day after buying
  $10 of credit once. A full evaluation needs at least 286 requests, so on the 50/day tier use `--limit`, or `--tasks`
  in daily batches.
- **Daily limit:** hitting it stops the evaluation and keeps partial results. Run `report` on that folder.
- **Settings:** `effort` is Anthropic-only and is ignored here. Consider `QRA_TEMPERATURE=0` for less variance.
  Usage and the provider-reported cost (0 for free models) are recorded.
- **Data:** free endpoints may log prompts. Only the public benchmark prompts and generated code are sent.

**Cost note:** a full `evaluate --mode both` makes at least 286 model calls (143 tasks × 2 modes), plus up to
2 repair calls per failing repair-mode task. Try `--limit` first.

**Resuming:** `--resume results/<eval_id>` takes the model, settings, tasks, modes and repeats from that
evaluation's `metadata.json`, not from the current command line, and skips runs already recorded. It refuses to
continue if the prompt version, dataset, task versions, sandbox image or harness changed. Typical use: an
OpenRouter free-tier run stops at the 50/day limit, and you resume it the next day.

**Dashboard:** `streamlit run dashboard/app.py` (optionally `-- path/to/results`). It is a read-only browser:
pick an evaluation to see headline metrics, the McNemar p-value, failure categories, a filterable run table,
and each attempt's code, error and traceback.

## Outputs

Each `run` or `evaluate` creates `results/<eval_id>/`:

| File | Content |
|---|---|
| `metadata.json` | Date, model, provider, config, dataset SHA-256, task ids and versions, prompt version, Docker/image/harness versions |
| `runs.jsonl` | One full record per (task, mode, repeat), including every attempt's code, execution result, tokens and timings. Appended as each run finishes, so partial evaluations stay usable |
| `runs.csv`, `attempts.csv` | Flat exports |
| `summary.json` | All metrics plus per-task rows |
| `report.md` | Human-readable comparison, failure categories, per-task table and limitations |

Results contain generated code and error text, never API keys. `results/` is gitignored.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Docker daemon is not reachable` | Start Docker Desktop. The tool refuses to run generated code without it |
| `Sandbox image ... not found` | `docker build -t qiskit-repair-runner:2.4.2 docker/` |
| `no validation manifest` / `not validated` | `validate-tasks`, or pass `--include-unvalidated` (not for reported results) |
| `fatal provider error: AuthenticationError` | Set `ANTHROPIC_API_KEY`. The evaluation stops and keeps partial results |
| `BadRequestError ... temperature` | Leave `temperature` unset for Opus 5.5. Use `effort` |
| Many `environment_error` attempts | The sandbox, not the model, is failing. Check the image build and Docker resources |

## Known limitations

- **Isolation is risk reduction, not a guarantee.** Docker shares the host kernel. A kernel or Docker
  escape would bypass every restriction. This is a local research tool, not a sandbox for hostile code.
  Generated code also shares the harness's interpreter, so deliberately malicious code could forge a
  "passed" result. Execution is contained, but honest grading assumes non-adversarial output.
- **Feedback includes assertion messages.** Hidden test *source* is never sent to the model, but the
  repair prompt contains the failing assertion's message (e.g. `Expected 3 qubits, got 2`). That is the
  intended experimental difference, but it means repair mode sees test information the baseline does not.
- **Excluded tasks.** 8 of 151 tasks need IBM Quantum cloud access and cannot run offline.
- **Failure categories are heuristic** (exception type and traceback origin). See `docs/architecture.md`.
- **Providers.** Anthropic and OpenRouter, behind a small `Provider` protocol (`llm.py`). Compare modes only within one model.
- **Cost is not reported.** The Messages API returns token usage but not cost, so cost is recorded as
  unknown rather than estimated.
- **Significance:** `report.md` gives an exact McNemar p-value on the discordant task pairs and the improvement per
  repeat. With repeats > 1, the pairs are not independent and the p-value is flagged as optimistic. Use the
  per-repeat spread, or test a single repeat.

See `docs/architecture.md` and `docs/evaluation_protocol.md`.
