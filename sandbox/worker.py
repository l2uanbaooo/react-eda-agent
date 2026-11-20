"""REPL-style kernel that runs INSIDE the sandbox (Docker container or subprocess).

Protocol: one JSON request per line on stdin; one response per line on the
ORIGINAL stdout, prefixed with "@@EDA@@ " (ASCII-only JSON). User prints are
captured per step; anything written straight to file descriptor 1 by C code or
child processes is redirected to stderr, so it can never corrupt the protocol.

Requests:
  {"op": "exec", "id": n, "code": "...", "timeout": 60}
  {"op": "get_answer", "id": n}
  {"op": "versions", "id": n}
  {"op": "ping", "id": n}
  {"op": "exit", "id": n}

The namespace persists across exec requests. Pre-defined names:
  pd, np, px, go, load_data, DATA_DIR, OUTPUT_DIR, save_fig(fig, name), answer=None
Soft timeout: SIGALRM raises StepTimeout (a BaseException, so `except Exception`
in user code cannot swallow it) and the namespace is kept. If that does not
work (signal ignored, stuck in C code) the host kills the whole worker.
"""
from __future__ import annotations

import contextlib
import io
import json
import linecache
import math
import os
import re
import signal
import sys
import time
import traceback
from pathlib import Path

PREFIX = "@@EDA@@ "
DATA_DIR = os.environ.get("EDA_DATA_DIR", "/data")
OUTPUT_DIR = os.environ.get("EDA_OUTPUT_DIR", "/output")
CODE_DIR = os.environ.get("EDA_CODE_DIR", "/opt/eda")
MAX_CHARS = int(os.environ.get("EDA_MAX_STDOUT_CHARS", "200000"))
FIG_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
HAS_ALARM = hasattr(signal, "SIGALRM")

# Keep a private copy of the real stdout for the protocol, then point fd 1 to stderr.
_proto = os.fdopen(os.dup(1), "w", encoding="ascii", errors="backslashreplace", buffering=1)
os.dup2(2, 1)
sys.stdout = io.TextIOWrapper(os.fdopen(1, "wb", buffering=0), encoding="utf-8", write_through=True)


class StepTimeout(BaseException):
    """Raised inside user code when the soft timeout expires."""


def _send(obj: dict) -> None:
    _proto.write(PREFIX + json.dumps(obj, ensure_ascii=True) + "\n")
    _proto.flush()


def _cap(s: str) -> str:
    if len(s) <= MAX_CHARS:
        return s
    half = MAX_CHARS // 2
    return s[:half] + f"\n... [{len(s) - MAX_CHARS} chars cut by sandbox] ...\n" + s[-half:]


def to_jsonable(v, depth: int = 0):
    """Convert numpy/pandas values to plain JSON types; raise TypeError if impossible."""
    if depth > 20:
        raise TypeError("answer is nested too deeply")
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    try:
        import numpy as np
        import pandas as pd
    except ImportError:  # pragma: no cover
        np = pd = None
    if np is not None:
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return to_jsonable(float(v))
        if isinstance(v, np.ndarray):
            return [to_jsonable(x, depth + 1) for x in v.tolist()]
    if pd is not None:
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        if isinstance(v, pd.Period):
            return str(v)
        if isinstance(v, pd.Series):
            return {str(k): to_jsonable(x, depth + 1) for k, x in v.items()}
        if isinstance(v, pd.DataFrame):
            return {str(c): to_jsonable(v[c], depth + 1) for c in v.columns}
        if v is pd.NaT:
            return None
    if isinstance(v, dict):
        return {str(k): to_jsonable(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [to_jsonable(x, depth + 1) for x in v]
    raise TypeError(f"answer of type {type(v).__name__} is not JSON-serializable")


def save_fig(fig, name: str) -> str:
    """Save a Plotly figure as OUTPUT_DIR/<name>.json (+ .html). Returns the json path."""
    if not FIG_NAME_RE.match(str(name)):
        raise ValueError("save_fig: name must match [A-Za-z0-9_-]{1,64}")
    if not hasattr(fig, "to_json") or not hasattr(fig, "write_html"):
        raise TypeError("save_fig expects a Plotly figure (plotly.express / graph_objects), "
                        f"got {type(fig).__name__}")
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / f"{name}.json"
    jpath.write_text(fig.to_json(), encoding="utf-8")
    fig.write_html(str(out / f"{name}.html"), include_plotlyjs="cdn", full_html=True)
    print(f"[save_fig] saved figure '{name}'")
    return str(jpath)


def _new_namespace() -> dict:
    sys.path.insert(0, CODE_DIR)
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from data import load_data

    return {"__name__": "__main__", "pd": pd, "np": np, "px": px, "go": go,
            "load_data": load_data, "save_fig": save_fig, "DATA_DIR": DATA_DIR,
            "OUTPUT_DIR": OUTPUT_DIR, "answer": None}


def _on_alarm(signum, frame):
    raise StepTimeout()


_step_counter = 0


def _exec(ns: dict, code: str, timeout: float) -> dict:
    global _step_counter
    _step_counter += 1
    fname = f"<step{_step_counter}>"
    # register the source so tracebacks show the failing line, not just its number
    linecache.cache[fname] = (len(code), None, code.splitlines(True), fname)
    out, err = io.StringIO(), io.StringIO()
    res = {"ok": True, "error": None, "traceback": None, "timed_out": False}
    t0 = time.perf_counter()
    if HAS_ALARM and timeout > 0:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            compiled = compile(code, fname, "exec")
            exec(compiled, ns)
    except StepTimeout:
        res.update(ok=False, timed_out=True,
                   error=f"TimeoutError: step exceeded {timeout:g} s and was interrupted")
    except BaseException as e:  # noqa: BLE001 - report everything, keep the kernel alive
        if isinstance(e, SystemExit):
            msg = "SystemExit: exit()/sys.exit() is not allowed in the sandbox"
        else:
            msg = f"{type(e).__name__}: {e}"
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        tb = [ln for ln in tb if 'sandbox/worker.py' not in ln and "worker.py" not in ln]
        res.update(ok=False, error=msg, traceback="".join(tb))
    finally:
        if HAS_ALARM:
            signal.setitimer(signal.ITIMER_REAL, 0)
    res.update(stdout=_cap(out.getvalue()), stderr=_cap(err.getvalue()),
               duration_s=round(time.perf_counter() - t0, 3))
    return res


def _versions() -> dict:
    import importlib.metadata as md

    pkgs = ["numpy", "pandas", "openpyxl", "pyarrow", "plotly", "scipy", "matplotlib",
            "scikit-learn", "seaborn"]
    out = {"python": sys.version.split()[0]}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def main() -> None:
    ns = _new_namespace()
    _send({"id": 0, "op": "ready", "ok": True, "pid": os.getpid(),
           "uid": os.getuid() if hasattr(os, "getuid") else None, "alarm": HAS_ALARM})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _send({"id": None, "ok": False, "error": f"bad request: {e}"})
            continue
        op, rid = req.get("op"), req.get("id")
        if op == "exec":
            _send({"id": rid, **_exec(ns, req.get("code", ""), float(req.get("timeout", 60)))})
        elif op == "get_answer":
            val = ns.get("answer")
            try:
                _send({"id": rid, "ok": True, "found": val is not None, "value": to_jsonable(val)})
            except (TypeError, ValueError) as e:
                _send({"id": rid, "ok": False, "found": True, "value": None, "error": str(e)})
        elif op == "versions":
            _send({"id": rid, "ok": True, "versions": _versions()})
        elif op == "ping":
            _send({"id": rid, "ok": True})
        elif op == "exit":
            _send({"id": rid, "ok": True})
            break
        else:
            _send({"id": rid, "ok": False, "error": f"unknown op {op!r}"})


if __name__ == "__main__":
    main()
