"""Run one task live, or launch a full evaluation in the background (results appear on the main page)."""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

from qiskit_repair_agent import runner as runner_mod
from qiskit_repair_agent.config import PROVIDERS, load_config, load_dotenv
from qiskit_repair_agent.controller import run_task
from qiskit_repair_agent.evaluator import new_eval_dir
from qiskit_repair_agent.llm import make_provider
from qiskit_repair_agent.models import ProviderError
from qiskit_repair_agent.reporting import append_record, write_exports
from qiskit_repair_agent.tasks import is_validated, load_tasks, load_validation

DEFAULT_MODELS = {"ollama": "qwen2.5-coder:7b", "openrouter": "nvidia/nemotron-3-super-120b-a12b:free",
                  "anthropic": "claude-opus-5-5", "mock": "reference"}

st.set_page_config(page_title="Run tasks", layout="wide")
load_dotenv()

with st.sidebar:
    provider = st.selectbox("Provider", PROVIDERS, index=PROVIDERS.index("ollama"))
    model = st.text_input("Model", DEFAULT_MODELS[provider], key=f"model-{provider}")
    dataset = st.selectbox("Dataset", ["standard", "hard"])
    max_repairs = st.number_input("Max repairs", 0, 10, 2)

try:
    cfg = load_config(None, provider=provider, model=model, max_repairs=int(max_repairs), dataset=dataset)
except ValueError as e:
    st.error(str(e))
    st.stop()

validation = load_validation(dataset)
tasks = [t for t in load_tasks(dataset) if is_validated(t, validation)]
single, full = st.tabs(["Single task", "Full evaluation"])

with single:
    task = st.selectbox("Task", tasks, format_func=lambda t: f"{t.task_id} · {t.difficulty} · {t.entry_point}")
    st.code(task.prompt, language="python")
    mode = st.radio("Mode", ["repair", "baseline"], horizontal=True)
    if st.button("Run", type="primary"):
        try:
            docker = runner_mod.DockerRunner(cfg)
            info = docker.check_available()
            llm = make_provider(cfg, [task])
            eval_id, out = new_eval_dir(cfg, label="run")
            with st.spinner(f"Running {task.task_id} with {llm.name}:{llm.model}..."):
                rec = run_task(task, mode, llm, docker, cfg, eval_id=eval_id)
        except (runner_mod.RunnerUnavailable, ProviderError, ValueError) as e:
            st.error(str(e))
            st.stop()
        append_record(out / "runs.jsonl", rec)
        write_exports([rec], out, {"eval_id": eval_id, "model": llm.model, "provider": llm.name,
                                   "dataset": dataset, "config": cfg.public_dict(), "versions": info})
        (st.success if rec.final_passed else st.error)(
            f"{rec.final_status} after {rec.repairs_used} repair(s) in {rec.total_latency_s:.1f}s · saved to {out}")
        for a in rec.attempts:
            e = a.execution
            with st.expander(f"Attempt {a.index} ({a.kind}) — {'PASS' if a.passed else a.category}",
                             expanded=a is rec.attempts[-1]):
                if a.llm_error:
                    st.error(a.llm_error)
                if a.code:
                    st.code(a.code, language="python")
                if e and not e.passed:
                    st.markdown(f"**{e.error_type}** at stage `{e.stage}`: {e.message}")

with full:
    st.caption(f"{len(tasks)} validated tasks. Runs the CLI in the background; a full run takes hours locally.")
    c1, c2, c3 = st.columns(3)
    eval_mode = c1.selectbox("Modes", ["both", "baseline", "repair"])
    repeats = c2.number_input("Repeats", 1, 10, 1)
    limit = c3.number_input("Limit (0 = all tasks)", 0, len(tasks), 0)
    proc = st.session_state.get("proc")
    if st.button("Start evaluation", disabled=proc is not None and proc.poll() is None):
        log = Path(cfg.results_dir) / "ui-logs" / f"{datetime.now():%Y%m%d-%H%M%S}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "qiskit_repair_agent.cli", "evaluate", "--provider", provider,
               "--model", model, "--dataset", dataset, "--max-repairs", str(max_repairs),
               "--mode", eval_mode, "--repeats", str(repeats)] + (["--limit", str(limit)] if limit else [])
        with open(log, "w", encoding="utf-8") as fh:
            st.session_state.proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        st.session_state.log = log
        proc = st.session_state.proc
    if proc is not None:
        running = proc.poll() is None
        st.write(f"**{'Running' if running else f'Finished (exit {proc.returncode})'}** · log: "
                 f"`{st.session_state.log}`")
        if running and st.button("Stop"):
            proc.terminate()
        st.button("Refresh")
        lines = Path(st.session_state.log).read_text(encoding="utf-8", errors="replace").splitlines()
        st.code("\n".join(lines[-40:]) or "(no output yet)", language="text")
