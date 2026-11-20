"""Common sandbox interface + the client that talks to sandbox/worker.py.

DockerSandbox and SubprocessSandbox differ only in how the worker process is
started (_command) and hard-killed (_kill_hard). Everything else - request
framing, soft/hard timeouts, restart after a crash, answer retrieval - lives
here once.

Timeouts:
  soft (cfg.step_timeout_s): the worker interrupts the step with SIGALRM and
       keeps all variables (Linux only: Docker / Kaggle).
  hard (soft + cfg.hard_timeout_grace_s): the host kills the worker, starts a
       fresh one and reports state_lost=True so the agent knows to reload data.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

PREFIX = b"@@EDA@@ "
WORKER_PATH = Path(__file__).resolve().parent / "worker.py"
DATA_PY_PATH = Path(__file__).resolve().parents[1] / "data.py"


class SandboxError(RuntimeError):
    """The sandbox could not be started or used (setup problem, not user code)."""


class WorkerDied(SandboxError):
    """The worker process exited unexpectedly (OOM kill, crash, killed)."""


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    traceback: str | None = None
    timed_out: bool = False
    state_lost: bool = False
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n... [{cut} characters truncated] ...\n{text[-tail:]}"


def short_traceback(tb: str | None, max_lines: int = 12) -> str:
    if not tb:
        return ""
    lines = tb.rstrip().splitlines()
    return "\n".join(lines[-max_lines:])


def format_observation(res: ExecResult, max_chars: int = 2000) -> str:
    """Text the LLM sees after a step (stdout, stderr, error), length-limited."""
    parts = []
    if res.stdout.strip():
        parts.append(res.stdout.rstrip())
    if res.stderr.strip():
        parts.append("[stderr]\n" + res.stderr.rstrip())
    if not res.ok:
        parts.append(f"[error] {res.error}")
        tb = short_traceback(res.traceback)
        if tb:
            parts.append(tb)
    if res.state_lost:
        parts.append("[sandbox] The Python session was restarted: all variables were lost. "
                     "Reload the data before continuing.")
    text = "\n".join(parts) if parts else "(no output)"
    return truncate_middle(text, max_chars)


class WorkerClient:
    """Owns one worker process and its stdin/stdout/stderr pipes."""

    def __init__(self, cmd: list[str], env: dict | None = None, cwd: str | None = None):
        self.cmd, self.env, self.cwd = cmd, env, cwd
        self.proc: subprocess.Popen | None = None
        self._responses: queue.Queue = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=200)
        self._next_id = 1
        self.ready_info: dict = {}

    def start(self, timeout: float) -> dict:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=self.env, cwd=self.cwd,
                                     creationflags=flags)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        msg = self._wait_for(0, timeout)
        if msg is None or msg.get("op") != "ready":
            self.kill()
            raise SandboxError(f"worker did not start within {timeout:g} s. "
                               f"stderr tail:\n{self.stderr_tail()}")
        self.ready_info = msg
        return msg

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for raw in iter(self.proc.stdout.readline, b""):
            if raw.startswith(PREFIX):
                try:
                    self._responses.put(json.loads(raw[len(PREFIX):].decode("ascii")))
                except (ValueError, UnicodeDecodeError):
                    self._stderr.append(f"[host] unparseable worker line: {raw[:200]!r}")
            else:
                self._stderr.append(raw.decode("utf-8", "replace").rstrip())
        self._responses.put({"__eof__": True})

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for raw in iter(self.proc.stderr.readline, b""):
            self._stderr.append(raw.decode("utf-8", "replace").rstrip())

    def stderr_tail(self, n: int = 30) -> str:
        return "\n".join(list(self._stderr)[-n:])

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _wait_for(self, rid: int, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            try:
                msg = self._responses.get(timeout=min(left, 0.5))
            except queue.Empty:
                if not self.alive():
                    raise WorkerDied(f"worker exited with code {self.proc.poll() if self.proc else None}")
                continue
            if msg.get("__eof__"):
                raise WorkerDied(f"worker exited with code {self.proc.wait() if self.proc else None}")
            if msg.get("id") == rid:
                return msg
            # a late answer to an older request: ignore

    def request(self, payload: dict, timeout: float) -> dict | None:
        """Send one request; return the response, or None if `timeout` expired."""
        if not self.alive():
            raise WorkerDied("worker is not running")
        rid = self._next_id
        self._next_id += 1
        line = json.dumps({**payload, "id": rid}, ensure_ascii=True) + "\n"
        try:
            assert self.proc and self.proc.stdin
            self.proc.stdin.write(line.encode("ascii"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise WorkerDied(f"cannot write to worker: {e}") from e
        return self._wait_for(rid, timeout)

    def kill(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


class Sandbox(ABC):
    """Persistent Python session for agent code. Use as a context manager."""

    name = "base"

    def __init__(self, cfg, data_dir: str | Path, output_dir: str | Path):
        self.cfg = cfg
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        if not self.data_dir.is_dir():
            raise SandboxError(f"data_dir does not exist: {self.data_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client: WorkerClient | None = None
        self.restarts = 0
        self.ready_info: dict = {}

    # ---- to implement ----
    @abstractmethod
    def _command(self) -> tuple[list[str], dict | None, str | None]:
        """(argv, env, cwd) that starts sandbox/worker.py."""

    @abstractmethod
    def _kill_hard(self) -> None:
        """Kill the worker (and its container) immediately."""

    @abstractmethod
    def data_path(self, relative: str) -> str:
        """Path of a file under data_dir as seen by code inside the sandbox."""

    def _preflight(self) -> None:
        """Checks before starting (e.g. Docker daemon reachable)."""

    # ---- lifecycle ----
    def start(self) -> "Sandbox":
        if self.client and self.client.alive():
            return self
        self._preflight()
        cmd, env, cwd = self._command()
        self.client = WorkerClient(cmd, env=env, cwd=cwd)
        self.ready_info = self.client.start(self.cfg.sandbox_start_timeout_s)
        return self

    def _restart(self) -> None:
        self._kill_hard()
        self.restarts += 1
        self.client = None
        self.start()

    def close(self) -> None:
        if self.client is None:
            return
        try:
            if self.client.alive():
                self.client.request({"op": "exit"}, timeout=3)
        except SandboxError:
            pass
        finally:
            self._kill_hard()
            self.client = None

    def __enter__(self) -> "Sandbox":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- operations ----
    def run(self, code: str, timeout: float | None = None) -> ExecResult:
        if self.client is None or not self.client.alive():
            self.start()
        soft = float(timeout if timeout is not None else self.cfg.step_timeout_s)
        hard = soft + self.cfg.hard_timeout_grace_s
        t0 = time.monotonic()
        try:
            resp = self.client.request({"op": "exec", "code": code, "timeout": soft}, timeout=hard)
        except WorkerDied:
            tail = self.client.stderr_tail(5) if self.client else ""
            self._restart()
            return ExecResult(ok=False, state_lost=True, duration_s=round(time.monotonic() - t0, 3),
                              error="SandboxCrash: the Python process died (out of memory or "
                                    f"killed); all variables were lost. {tail}".strip())
        if resp is None:
            self._restart()
            return ExecResult(ok=False, timed_out=True, state_lost=True,
                              duration_s=round(time.monotonic() - t0, 3),
                              error=f"TimeoutError: step exceeded {soft:g} s and could not be "
                                    "interrupted; the sandbox was restarted and all variables were lost")
        return ExecResult(ok=bool(resp.get("ok")), stdout=resp.get("stdout", ""),
                          stderr=resp.get("stderr", ""), error=resp.get("error"),
                          traceback=resp.get("traceback"), timed_out=bool(resp.get("timed_out")),
                          duration_s=float(resp.get("duration_s", 0.0)))

    def get_answer(self) -> tuple[bool, object, str | None]:
        """(found, value, error). found=False when `answer` is still None."""
        if self.client is None or not self.client.alive():
            return False, None, "sandbox not running"
        resp = self.client.request({"op": "get_answer"}, timeout=30)
        if resp is None:
            return False, None, "timeout reading answer"
        if not resp.get("ok"):
            return bool(resp.get("found")), None, resp.get("error")
        return bool(resp.get("found")), resp.get("value"), None

    def versions(self) -> dict:
        if self.client is None or not self.client.alive():
            self.start()
        resp = self.client.request({"op": "versions"}, timeout=60)
        return (resp or {}).get("versions", {})

    def list_outputs(self) -> list[str]:
        return sorted(p.name for p in self.output_dir.iterdir() if p.is_file())

    def info(self) -> dict:
        return {"sandbox": self.name, "restarts": self.restarts,
                "uid": self.ready_info.get("uid"), "soft_timeout": self.ready_info.get("alarm")}
