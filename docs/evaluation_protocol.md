# Evaluation protocol

## Research question

Does execution-guided self-repair raise the Qiskit HumanEval pass rate over one-shot generation with
the same model, and what extra latency and tokens does it cost? No answer is assumed. This repository
contains no real-model results yet.

## Benchmark

- Source: official Qiskit HumanEval (`qiskit-community/qiskit-human-eval`, Apache-2.0), vendored
  unmodified. See `src/qiskit_repair_agent/benchmark/NOTICE.md` for hashes.
- Two variants: **standard** (signature + docstring stub; HumanEval-style) and **hard** (prose only).
  Report them separately.
- Task version = SHA-256 prefix of prompt, prelude, test and entry point. Records store it, and a task
  whose content changes no longer counts as validated.
- **Validation** (`validate-tasks`): in the same sandbox and limits used for evaluation, the canonical
  solution must pass and a `return None` stub must fail. Result on 2026-10-18: 143/151 usable in both
  variants. Excluded: 43, 97, 98, 122, 129, 133, 134, 146, which need `QiskitRuntimeService` (IBM
  Quantum account or network). Re-run validation whenever the image or harness changes.
- Pinned environment: Python 3.12 image with `qiskit 2.4.2`, `qiskit-aer 0.17.0`,
  `qiskit-ibm-runtime 0.45.0` and others from the benchmark's own `requirements.txt`.

## Fairness controls

Identical across modes: task ids and versions, system and initial prompts (`PROMPT_VERSION=v1`), model,
generation settings, sandbox image and limits, tests. The only intended difference: repair mode
receives the failure summary and previous code, and may generate up to `max_repairs` more attempts.

- Each mode makes its **own** initial generation. The initial pass rates of the two modes therefore
  estimate the same quantity, and their difference (`initial_pass_rate_difference_pp`) shows
  model nondeterminism. Use `--repeats` to quantify it.
- Modes are interleaved per task (baseline, then repair, for task 1, and so on) so that API drift
  affects both equally.
- Do not edit prompts, tasks, the harness or the config during an evaluation. If prompt wording
  changes, bump `PROMPT_VERSION`.
- Develop against a subset (`--limit`, `--tasks`). Treat the full run as the final evaluation.
  Because the benchmark is fixed, a separate held-out split is not available. Disclose that any
  prompt tuning used the same tasks.

## Metrics (per mode, from `reporting.compute_metrics`)

Denominators:

- **all runs**: every (task, repeat). Provider failures count as not passed. Nothing is dropped.
- **valid runs**: excludes `provider_error`, `blocked` and `internal_error`, where no solution was
  evaluated. Both are reported.

| Metric | Definition |
|---|---|
| First-attempt pass rate | Runs whose attempt 0 passed / runs |
| Final pass rate | Runs passing after all allowed attempts / runs |
| Absolute improvement | Final pass rate (repair) − final pass rate (baseline), in percentage points |
| Repair success rate | Among valid repair-mode runs failing at attempt 0, the fraction that finally pass |
| Repair attempts | Mean, mean over initially failing runs, and the distribution of `repairs_used` |
| Latency | Mean and median end-to-end seconds per run (LLM + sandbox) |
| Usage | Total input/output tokens. Cost is "unknown" because the API does not report it |
| Failure distribution | Categories over all failing attempts, initial attempts, and final attempts |
| Paired table | Per (task, repeat): both pass / repair only / baseline only / neither |

## Reporting rules

- State the task count, excluded tasks, model id, effort, `max_repairs`, repeats, dataset variant,
  date, image id and package versions. `metadata.json` and `report.md` contain them.
- Report absolute differences in percentage points. Report the paired discordant counts
  (repair_only vs baseline_only). With repeats, report the spread across repeats. Do not claim
  significance without a test that supports it. `report.md` includes the exact McNemar p-value on the
  discordant pairs. It is valid for a single repeat; with repeats > 1 it is flagged as optimistic.
- A single run is a single sample of a stochastic model. Say so.
- Ablation over 0–3 repairs: `--max-repairs N` with `--mode repair`. `--max-repairs 0` reproduces the
  baseline behaviour inside repair mode.

## Reproduction

```bash
docker build -t qiskit-repair-runner:2.4.2 docker/
python -m qiskit_repair_agent.cli validate-tasks                       # confirm 143/151 in your environment
python -m qiskit_repair_agent.cli evaluate --mode both --config configs/default.toml --repeats 3
python -m qiskit_repair_agent.cli report
```
