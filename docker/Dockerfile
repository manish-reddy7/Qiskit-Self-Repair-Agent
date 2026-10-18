# Sandbox image for executing LLM-generated Qiskit code.
# Build: docker build -t qiskit-repair-runner:2.4.2 docker/
FROM python:3.12-slim

# graphviz: needed by the DAG/visualisation tasks (qiskitHumanEval/29, /123).
RUN apt-get update \
 && apt-get install -y --no-install-recommends graphviz \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-runner.txt /tmp/requirements-runner.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runner.txt \
 && rm /tmp/requirements-runner.txt \
 && python -c "import qiskit, qiskit_aer, qiskit_ibm_runtime; print(qiskit.__version__)"

# Unprivileged user; the host runner also forces --user 10001:10001.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin runner
USER 10001
WORKDIR /tmp
ENV HOME=/tmp MPLCONFIGDIR=/tmp MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
