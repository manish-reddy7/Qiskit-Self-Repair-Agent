"""Executes generated code in a locked-down Docker container. There is deliberately no host fallback."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from . import harness
from .config import Config
from .models import ExecResult, Task

HARNESS_SOURCE = Path(harness.__file__).read_text(encoding="utf-8")
HARNESS_SHA256 = hashlib.sha256(HARNESS_SOURCE.encode()).hexdigest()


class RunnerUnavailable(RuntimeError):
    pass


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


class DockerRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def check_available(self) -> dict:
        """Raise RunnerUnavailable with an actionable message, else return version info."""
        if shutil.which("docker") is None:
            raise RunnerUnavailable("Docker CLI not found. Install Docker Desktop/Engine; generated code is never "
                                    "run on the host.")
        info = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                              capture_output=True, text=True, timeout=30)
        if info.returncode != 0:
            raise RunnerUnavailable("Docker daemon is not reachable (is Docker Desktop running?): "
                                    + info.stderr.strip()[:300])
        image = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", self.cfg.image],
                               capture_output=True, text=True, timeout=30)
        if image.returncode != 0:
            raise RunnerUnavailable(f"Sandbox image {self.cfg.image!r} not found. Build it with:\n"
                                    f"  docker build -t {self.cfg.image} docker/")
        return {"docker_server": info.stdout.strip(), "image": self.cfg.image, "image_id": image.stdout.strip(),
                "harness_sha256": HARNESS_SHA256}

    def command(self, name: str) -> list[str]:
        c = self.cfg
        return [
            "docker", "run", "--rm", "-i", "--name", name,
            "--network", "none",
            "--user", "10001:10001",
            "--read-only",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m",
            "--memory", c.memory, "--memory-swap", c.memory,
            "--cpus", str(c.cpus),
            "--pids-limit", str(c.pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--ipc", "private", "--shm-size", "64m",   # qiskit's parallel_map needs /dev/shm semaphores
            c.image, "python", "-I", "-c", HARNESS_SOURCE,
        ]

    def _kill(self, name: str) -> None:
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)

    def _preflight(self) -> None:
        if shutil.which("docker") is None:
            raise RunnerUnavailable("Docker CLI not found; refusing to execute generated code on the host.")

    def execute(self, task: Task, code: str) -> ExecResult:
        self._preflight()
        payload = json.dumps({"prelude": task.prelude, "code": code, "test": task.test,
                              "entry_point": task.entry_point, "max_chars": self.cfg.max_output_chars})
        name = f"qra-{uuid.uuid4().hex[:12]}"
        start = time.monotonic()
        try:
            proc = subprocess.run(self.command(name), input=payload, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=self.cfg.exec_timeout_s)
        except subprocess.TimeoutExpired as e:
            self._kill(name)
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return ExecResult(passed=False, category="timeout", stage="sandbox", error_type="Timeout",
                              message=f"execution exceeded {self.cfg.exec_timeout_s:g}s and was killed",
                              output=_truncate(out, self.cfg.max_output_chars), timed_out=True,
                              duration_s=time.monotonic() - start)
        duration = time.monotonic() - start
        return parse_harness_output(proc.stdout, proc.stderr, proc.returncode, duration, self.cfg.max_output_chars)


def parse_harness_output(stdout: str, stderr: str, exit_code: int, duration: float, limit: int) -> ExecResult:
    line = next((ln for ln in reversed(stdout.splitlines()) if ln.startswith(harness.MARKER)), None)
    if line is None:
        # No result line: the process died (OOM kill = 137), called os._exit, or the sandbox failed to start.
        if exit_code == 125:   # docker itself failed (daemon gone, bad image): the code never ran
            category, msg = "blocked", "docker could not start the sandbox container: " + stderr.strip()[:300]
        elif exit_code in (126, 127):
            category, msg = "environment_error", "sandbox container could not run the harness"
        elif exit_code == 137:
            category, msg = "resource_error", "process killed (exit 137, likely the memory limit)"
        else:
            category, msg = "runtime_error", f"program exited with code {exit_code} before the tests completed"
        return ExecResult(passed=False, category=category, stage="sandbox", error_type="NoResult", message=msg,
                          output=_truncate(stdout + stderr, limit), exit_code=exit_code, duration_s=duration)
    data = json.loads(line[len(harness.MARKER):])
    return ExecResult(passed=data["passed"], category=data["category"], stage=data["stage"],
                      error_type=data["error_type"], message=data["message"],
                      traceback=_truncate(data["traceback"], limit), output=data["output"],
                      exit_code=exit_code, duration_s=duration, qiskit_version=data["qiskit_version"])
