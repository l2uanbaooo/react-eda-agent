"""SubprocessSandbox (mode B, fallback): the worker runs as a plain child process.

NO isolation: no network block, no read-only data, no memory limit. It exists
only for running everything inside a Kaggle notebook, where Docker is not
available. Results produced with it must be labelled sandbox=subprocess.
Soft timeouts need SIGALRM (Linux/Kaggle); on Windows only the hard timeout
works, so a timed-out step always restarts the session there.
"""
from __future__ import annotations

import os
import sys

from .base import DATA_PY_PATH, WORKER_PATH, Sandbox


class SubprocessSandbox(Sandbox):
    name = "subprocess"

    def _command(self):
        env = dict(os.environ)
        env.update({
            "EDA_DATA_DIR": str(self.data_dir),
            "EDA_OUTPUT_DIR": str(self.output_dir),
            "EDA_CODE_DIR": str(DATA_PY_PATH.parent),
            "EDA_MAX_STDOUT_CHARS": str(self.cfg.max_stdout_chars),
            "EDA_DEFAULT_SHEET": self.cfg.sandbox_default_sheet,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
        })
        return [sys.executable, "-u", str(WORKER_PATH)], env, str(self.output_dir)

    def _kill_hard(self) -> None:
        if self.client:
            self.client.kill()

    def data_path(self, relative: str) -> str:
        return str(self.data_dir / relative)
