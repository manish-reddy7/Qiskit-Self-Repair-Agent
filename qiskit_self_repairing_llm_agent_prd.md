# LLM Assistant That Writes and Self-Repairs Qiskit Code

**Product Requirements Document (PRD)**  
**Version:** 1.0 · **Status:** Draft for implementation · **Project type:** Academic research prototype  
**Source brief:** Quantum Computing (25AM3504), Project 7 — *LLM assistant that writes and self-repairs Qiskit code*  
**Assigned base paper:** *Qiskit HumanEval: An Evaluation Benchmark for Quantum Code Generative Models*

## 1. Executive summary

Build a local-first research prototype that accepts a natural-language quantum programming task, asks an LLM to generate Qiskit code, runs the code in a controlled environment, inspects test or execution failures, and feeds concise error feedback to the LLM for a limited number of repair attempts. The product must compare this self-repair workflow with a baseline that makes a single generation attempt without repair.

The primary output is not merely a code snippet. It is a traceable result containing the generated code, execution and test results, repair history, and benchmark metrics. The prototype is intended for controlled evaluation and learning, not for executing untrusted code in a production environment.

## 2. Problem statement

LLMs can generate quantum-programming code that looks plausible but may fail because of incorrect Qiskit APIs, invalid circuit construction, measurement mistakes, mismatched expected behavior, or ordinary Python errors. A one-shot generation workflow gives limited visibility into whether the code actually works. A repair loop may improve task completion, but its value must be measured against a comparable baseline rather than assumed.

**Problem to solve:** Can an LLM that runs generated Qiskit code and receives structured failure feedback solve more benchmark tasks than the same model used in a one-shot baseline, at an acceptable cost and latency?

## 3. Goals and non-goals

### Goals
- Generate Qiskit Python code from a clearly specified quantum-programming task.
- Execute generated code safely in an isolated, resource-limited environment.
- Validate solutions using predefined tests and observable execution results.
- Support a configurable, bounded self-repair loop using error feedback.
- Evaluate baseline versus self-repair performance on a fixed benchmark suite.
- Record reproducible run details, failure categories, latency, and estimated model usage/cost where available.
- Provide a simple local interface for running a task and inspecting results.

### Non-goals
- A general-purpose autonomous coding agent for arbitrary repositories or languages.
- Execution of generated code on real quantum hardware in the MVP.
- A guarantee of mathematical or quantum-physical correctness beyond the implemented tests and checks.
- Automatic modification of a user's existing project or files.
- Training or fine-tuning a foundation model.
- Claims that self-repair is better before the experiment is run.

## 4. Target users and use cases

**Primary user — student/researcher:** wants to study LLM reliability on Qiskit coding tasks and compare a repair loop with a one-shot baseline.

**Secondary user — developer/learner:** wants to submit a small Qiskit task, inspect generated code, and understand why it failed.

### Core user journey
1. Select or enter a benchmark task.
2. Choose the model configuration and run mode: **Baseline** or **Self-repair**.
3. The model generates Qiskit code.
4. The runner executes the code under time and resource limits and applies the task's tests.
5. In self-repair mode, structured errors and failing test summaries are sent back to the model until success or the repair limit is reached.
6. The user sees the final code, pass/fail outcome, run trace, duration, and benchmark record.

## 5. Scope and release plan

### MVP — required
- Python-based task format for benchmark prompts and tests.
- LLM adapter with model/provider settings externalized to environment configuration.
- One-shot generation baseline.
- Bounded self-repair workflow.
- Isolated execution runner with timeouts and output limits.
- Automated correctness checks for each benchmark task.
- Run logs and machine-readable result export (JSON or CSV).
- Comparison report for baseline versus self-repair.
- Local CLI; a minimal Streamlit interface is optional if time allows.

### Post-MVP — optional
- Web dashboard with run history and charts.
- Additional models and prompt strategies.
- Error-category analysis and ablation experiments, such as 0, 1, 2, and 3 repair attempts.
- More benchmark tasks or a held-out task set.
- Optional simulator comparison or hardware execution only if access and budget permit.

### Explicitly out of scope for the first release
- Multi-agent orchestration.
- Arbitrary shell commands, network access, or unrestricted file-system access by generated code.
- Persistent autonomous execution without user initiation.
- Claims of production-grade sandbox security.

## 6. Functional requirements

| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-01 | Task input and task selection | Must | User can select a packaged benchmark task or enter a task prompt with a defined expected behavior. Invalid/empty tasks are rejected with a clear message. |
| FR-02 | LLM code generation | Must | System produces a Python/Qiskit code artifact and records model/configuration identifiers and prompt version. |
| FR-03 | Baseline mode | Must | Baseline generates and evaluates the code once; it does not call the repair loop. |
| FR-04 | Execution and validation | Must | Code runs in an isolated worker with a timeout, bounded output, and a task-specific test result. Timeout and execution errors are represented as structured outcomes. |
| FR-05 | Self-repair loop | Must | On failure, system sends a concise error/test summary and the previous code to the model; retries stop on passing tests, the configured maximum attempts, or a hard resource limit. |
| FR-06 | Run trace | Must | Each run records generated versions, errors, repair attempt count, test outcomes, durations, and final status. Secrets are never written to logs. |
| FR-07 | Evaluation | Must | A repeatable evaluation command runs both modes on the same task set and outputs aggregate and per-task metrics. |
| FR-08 | Result export | Must | Results can be exported to JSON or CSV with stable task/run identifiers. |
| FR-09 | Configuration | Should | Model/provider, API endpoint if relevant, temperature, timeout, max repair attempts, and output limits can be changed without source-code edits. Secrets are read from environment variables. |
| FR-10 | User interface | Should | CLI shows progress and a readable final summary. A simple local web UI can be added without changing core logic. |
| FR-11 | Failure classification | Should | Failures are classified where possible as syntax/import, API misuse, runtime, timeout/resource, test failure, or model/API failure. |
| FR-12 | Reproducibility | Should | Task set version, prompt version, model identifier, configuration, and software versions are saved with each evaluation. |

## 7. Non-functional requirements

- **Safety:** Execute generated code in a separate process/container with no network access, restricted file access, CPU/memory/time limits, and an unprivileged user. Treat sandboxing as a risk reduction, not a proof of perfect isolation.
- **Reliability:** One task failing must not terminate the full benchmark run. Individual task failures should be captured and the evaluation should continue.
- **Performance:** Initial target is a configurable 30-second execution timeout per attempt; actual model latency is measured, not guaranteed. Default maximum repair attempts: 2, excluding the initial generation.
- **Reproducibility:** The same task set and configuration must be runnable again, and result records must preserve relevant version/configuration metadata.
- **Maintainability:** Separate prompting, model access, execution, testing, evaluation, and UI layers.
- **Usability:** A new user should be able to run a sample task from the README with clearly documented setup steps.
- **Cost visibility:** Record token usage and/or provider-reported cost when available; mark unavailable values as unknown rather than fabricating estimates.
- **Privacy:** Do not store API keys in source control. Avoid sending unrelated local files or personal information to the model.

## 8. Proposed workflow and architecture

### Workflow
`Task / Benchmark → Prompt Builder → LLM Adapter → Code Artifact → Isolated Qiskit Runner → Task Tests → Result Evaluator`

On a failed run in self-repair mode:
`Structured Failure Summary + Previous Code → Repair Prompt → LLM Adapter → New Code Artifact → Runner → Tests`

The loop ends when tests pass, the retry budget is exhausted, a global budget is reached, or the run is cancelled.

### Logical components
- **Task Registry:** loads task ID, prompt, tests, expected output/behavior, tags, and task version.
- **Prompt Builder:** formats the initial request and repair request consistently; keeps prompt templates versioned.
- **LLM Adapter:** provider-independent interface for generation; handles retries for transient API errors and records usage where available.
- **Repair Controller:** manages attempts, stop conditions, failure summaries, and budget limits.
- **Execution Worker:** runs code in an isolated process/container, captures stdout/stderr, exit code, runtime, and timeout status.
- **Test Harness:** runs task-specific assertions and returns structured pass/fail details. Tests must not rely only on whether the program exits successfully.
- **Evaluator:** aggregates task outcomes and compares baseline against self-repair.
- **Result Store/Exporter:** saves JSONL/JSON or CSV run records and summary reports.
- **CLI / Optional UI:** starts runs and presents code, attempt history, and metrics.

## 9. Evaluation plan

### Experimental design
- Use a fixed, versioned benchmark based on the assigned Qiskit HumanEval task source where access and licensing permit. If that benchmark cannot be used directly, document the alternative task set and its limitations.
- Use the same tasks, model, generation settings, and test definitions for baseline and self-repair comparisons.
- Keep benchmark task tests hidden from the generation prompt when appropriate, to reduce test-specific overfitting.
- Run at least one evaluation pass across all selected tasks; repeat stochastic runs where budget allows and report the run count.
- Separate task development/debugging from final evaluation tasks where feasible.

### Primary metrics
- **Pass@1 / first-attempt pass rate:** share of tasks that pass after the initial generation.
- **Final pass rate:** share of tasks that pass after the allowed repair attempts.
- **Repair success rate:** among tasks that fail initially, share that subsequently pass within the repair budget.
- **Mean repair attempts:** average number of additional generation attempts.
- **Latency:** median and mean end-to-end time per task, reported separately for each mode.
- **Cost/usage:** total and per-task token usage or provider-reported cost when available.
- **Failure distribution:** counts by syntax/import, API misuse, runtime, timeout/resource, test failure, and model/API failure.

### Comparison and reporting rules
- Report absolute pass-rate difference in percentage points, not just relative improvement.
- Include task count, exclusions, model identifier, parameters, package versions, retry limit, and date of the run.
- Use per-task paired comparisons because the same tasks are evaluated in both modes.
- If repeated runs are affordable, report variability or confidence intervals; otherwise disclose that results come from a single run.
- Do not claim statistical significance unless the analysis supports it.

### Provisional MVP success criteria
These are project targets, not guaranteed results:
1. A minimum of 20 validated tasks is available in the benchmark suite; aim for 30 or more if practical.
2. Both modes run on the same task set and generate machine-readable results.
3. At least 95% of evaluation tasks produce a recorded terminal outcome, even if that outcome is failure or timeout.
4. Every repair attempt and its outcome is traceable for debugging.
5. Final report compares first-attempt and final pass rates, repair success, latency, and model usage/cost where available.
6. The README enables a teammate to set up the environment and reproduce a sample evaluation.

## 10. Data model and logging

Each evaluation record should include:
- `run_id`, `task_id`, `task_version`, `mode`, `model_id`, `prompt_version`, timestamp.
- Configuration: temperature, timeout, max repair attempts, runner/software versions.
- Attempt entries: generated code or code hash, attempt index, exit status, test summary, error category, duration, token usage if available.
- Final outcome: passed/failed/timed out/blocked/API error, number of attempts, end-to-end latency, total usage/cost if available.

Do not store credentials, environment secrets, or unrelated user files. If storing full prompts and code, document that behavior in the README and allow local result deletion.

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Generated code can be unsafe | Host data or process could be exposed | Isolated runner, no network, limited filesystem, unprivileged process, resource limits, no real secrets in the worker. |
| Self-repair overfits to error messages | Apparent improvement may not generalize | Use fixed held-out tasks where feasible, stable tests, and report per-task results. |
| Model/API availability or rate limits | Evaluation becomes incomplete or expensive | Add bounded retries/backoff, save progress incrementally, support a configurable provider, and report skipped tasks. |
| Qiskit version mismatch | Generated code or tests fail for environmental reasons | Pin dependencies, record versions, distinguish environment failures from solution failures. |
| Unfair baseline comparison | Misleading conclusions | Use identical model and generation settings where comparable; document the only intended difference as the repair loop. |
| Simulation time/resource spikes | Long or stalled runs | Per-attempt timeout, memory/CPU limits, maximum output, and run-level budget. |
| Small benchmark | Weak generalization claims | State sample size and limitations; avoid broad claims. |

## 12. MVP acceptance checklist

- [ ] Project installs from documented dependency files.
- [ ] One example Qiskit task runs end-to-end locally.
- [ ] Baseline mode generates once, executes, tests, and records the result.
- [ ] Self-repair mode retries only within the configured limit and stops on success.
- [ ] Execution timeout and runtime errors are captured without crashing the whole benchmark.
- [ ] Generated code runs only in the restricted worker, not directly in the main app process.
- [ ] Results are exportable in JSON/CSV and include the required metadata.
- [ ] Evaluation compares the same task IDs across modes.
- [ ] Summary report includes metrics, failure categories, limitations, and reproduction instructions.
- [ ] API keys and secrets are absent from source control and logs.

## 13. Proposed implementation milestones

| Phase | Work | Exit deliverable |
|---|---|---|
| 1. Foundation | Review benchmark source, choose task subset, define task schema, pin Qiskit and test dependencies | Task registry and documented evaluation protocol |
| 2. Baseline | Add model adapter, initial code-generation prompt, isolated runner and tests | Baseline evaluates sample tasks end-to-end |
| 3. Self-repair | Add structured error feedback, retry controller, stop conditions and run trace | Self-repair mode operates with bounded retries |
| 4. Evaluation | Run both modes, export results, calculate metrics and inspect failure categories | Reproducible comparison report and tables/plots |
| 5. Polish | README, setup scripts, configuration examples, demo and limitations | Portfolio-ready repository and short demo |

## 14. Suggested repository structure

```text
qiskit-self-repair-agent/
├── README.md
├── pyproject.toml                 # or requirements.txt
├── .env.example                   # placeholders only; no real secrets
├── configs/
│   └── default.yaml
├── tasks/
│   ├── registry.py
│   └── benchmark/                 # task definitions and tests
├── src/
│   ├── llm_adapter.py
│   ├── prompt_builder.py
│   ├── repair_controller.py
│   ├── runner.py
│   ├── test_harness.py
│   ├── evaluator.py
│   └── result_store.py
├── scripts/
│   ├── run_task.py
│   └── evaluate_benchmark.py
├── tests/
├── results/                       # local outputs; avoid committing sensitive data
└── docs/
    ├── evaluation_protocol.md
    └── architecture.md
```

This structure is a suggested starting point and may be adjusted to fit team size and implementation choices.

## 15. Suggested roles for a team of 3–4

- **LLM and prompting:** model adapter, initial/repair prompts, usage logging.
- **Execution and testing:** sandbox runner, timeout/resource controls, test harness.
- **Evaluation and research:** task curation, baseline protocol, metrics, analysis.
- **Integration and product (if 4 members):** CLI/UI, result export, documentation, integration tests.

For a three-person team, combine integration/documentation with evaluation or LLM work, but assign one owner to the safety boundary of the runner.

## 16. Key decisions to finalize before implementation

1. Which LLM provider/model will be used, and is there a spending limit?
2. Can the team access and legally use the assigned Qiskit HumanEval benchmark, or is a documented subset/alternative required?
3. Will code execution use Docker or another isolated runner available in the development environment?
4. How many repair attempts will the main experiment allow? Recommended MVP default: two additional attempts.
5. Is the CLI sufficient for delivery, or is a simple Streamlit UI required by the course/demo rubric?

## 17. Source basis and assumptions

**Directly from the assignment brief:** Project 7 asks for an LLM to be evaluated on Qiskit HumanEval tasks, with and without a loop that runs generated code and feeds errors back. The listed base paper is *Qiskit HumanEval: An Evaluation Benchmark for Quantum Code Generative Models*.

**Product design recommendations added in this PRD:** bounded retry limits, isolation, metrics, MVP acceptance criteria, repository layout, milestones, and suggested implementation choices are recommendations for making the assignment measurable and safer to implement. They are not additional requirements quoted from the assignment brief.
