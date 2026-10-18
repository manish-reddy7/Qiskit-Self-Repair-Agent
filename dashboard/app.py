"""Read-only results browser. Run: streamlit run dashboard/app.py  (reads results/ by default)."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from qiskit_repair_agent.reporting import compute_metrics, load_records

st.set_page_config(page_title="Qiskit self-repair results", layout="wide")
results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
evals = sorted((p.parent for p in results_dir.glob("*/runs.jsonl")), key=lambda p: p.stat().st_mtime, reverse=True)
if not evals:
    st.info(f"No results under `{results_dir}/`. Run `python -m qiskit_repair_agent.cli evaluate` first.")
    st.stop()

chosen = st.sidebar.selectbox("Evaluation", evals, format_func=lambda p: p.name)
meta_path = chosen / "metadata.json"
meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
records = load_records(chosen / "runs.jsonl")
metrics = compute_metrics(records)

st.title(chosen.name)
if meta.get("provider") == "mock":
    st.warning("Mock provider: scripted pipeline check, not an LLM result.")
if meta.get("aborted"):
    st.error(f"Evaluation stopped early: {meta['aborted']}. Resume with `evaluate --resume {chosen}`.")
st.caption(" · ".join(f"**{k}**: {meta[k]}" for k in ("model", "provider", "dataset", "prompt_version",
                                                      "max_repairs", "repeats", "n_tasks", "date") if k in meta))

if "comparison" in metrics:
    c, modes = metrics["comparison"], metrics["modes"]
    cols = st.columns(4)
    cols[0].metric("Baseline final pass", f"{100 * modes['baseline']['final_pass_rate']:.1f}%")
    cols[1].metric("Repair final pass", f"{100 * modes['repair']['final_pass_rate']:.1f}%",
                   f"{c['absolute_improvement_pp']:+.1f} pp")
    rs = modes["repair"]["repair_success_rate"]
    cols[2].metric("Repair success rate", "n/a" if rs is None else f"{100 * rs:.1f}%")
    p = c["mcnemar_exact_p"]
    cols[3].metric("McNemar exact p", "n/a" if p is None else f"{p:.3g}")
    st.write("Paired outcomes:", c["paired"])

st.subheader("Metrics by mode")
scalar = {m: {k: v for k, v in d.items() if not isinstance(v, dict)} for m, d in metrics["modes"].items()}
st.dataframe(pd.DataFrame(scalar), width="stretch")

st.subheader("Failure categories (all failing attempts)")
fail = pd.DataFrame({m: d["failure_distribution_all_attempts"] for m, d in metrics["modes"].items()}).fillna(0)
if not fail.empty:
    st.bar_chart(fail)

st.subheader("Runs")
runs = pd.DataFrame([{"task": r.task_id, "mode": r.mode, "repeat": r.repeat, "status": r.final_status,
                      "repairs": r.repairs_used, "latency_s": round(r.total_latency_s, 1),
                      "tokens_out": r.output_tokens, "run_id": r.run_id} for r in records])
status = st.multiselect("Status", sorted(runs["status"].unique()), default=sorted(runs["status"].unique()))
st.dataframe(runs[runs["status"].isin(status)], width="stretch", hide_index=True)

st.subheader("Attempt trace")
by_id = {r.run_id: r for r in records}
run_id = st.selectbox("Run", runs["run_id"], format_func=lambda i: f"{by_id[i].task_id} · {by_id[i].mode} · "
                      f"rep {by_id[i].repeat} · {by_id[i].final_status}")
for a in by_id[run_id].attempts:
    with st.expander(f"Attempt {a.index} ({a.kind}) — {'PASS' if a.passed else a.category}", expanded=not a.passed):
        st.caption(f"model {a.llm_model} · tokens {a.input_tokens}/{a.output_tokens} · {a.duration_s:.1f}s"
                   + (f" · retry after {a.retry_reason}" if a.retry_reason else ""))
        if a.llm_error:
            st.error(a.llm_error)
        if a.code:
            st.code(a.code, language="python")
        e = a.execution
        if e and not e.passed:
            st.markdown(f"**{e.error_type}** at stage `{e.stage}`: {e.message}")
            if e.traceback:
                st.code(e.traceback, language="text")
        if e and e.output.strip():
            st.text_area("Program output", e.output, height=120, key=f"out-{run_id}-{a.index}")
