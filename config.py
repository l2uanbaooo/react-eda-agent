"""The ONE configuration file of the project.

Every field can be overridden with an environment variable named EDA_<FIELD>
(upper case), e.g. in PowerShell:
    $env:EDA_OLLAMA_URL = "https://xxxx.trycloudflare.com"
    $env:EDA_SANDBOX_MODE = "subprocess"
Switching mode A (local Docker + Kaggle LLM) / mode B (all on Kaggle) is only
a matter of EDA_SANDBOX_MODE and EDA_OLLAMA_URL.

config_hash() is written into every log line so results can be traced back
to the exact settings that produced them.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SANDBOX_MODES = ("docker", "subprocess")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_TYPES = {"str": str, "int": int, "float": float, "bool": bool}
# Fields that never change what the agent does (where things are / how to reach them).
INFRA_FIELDS = {"ollama_url", "datasets_dir", "runs_dir", "llm_connect_timeout_s", "llm_read_timeout_s",
                "llm_max_retries", "llm_backoff_base_s", "sandbox_start_timeout_s",
                "eval_outage_wait_s", "eval_outage_checks"}


@dataclass(frozen=True)
class Config:
    # ---- LLM: Ollama over HTTP (mode A: the Kaggle tunnel URL) ----
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:14b"
    judge_model: str = "qwen2.5:14b-instruct"
    temperature: float = 0.1
    judge_temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192
    num_predict: int = 1024
    keep_alive: str = "24h"
    llm_connect_timeout_s: float = 10.0
    llm_read_timeout_s: float = 120.0      # max silence between streamed chunks
    llm_max_retries: int = 4               # retries after the first attempt
    llm_backoff_base_s: float = 1.0        # waits 1, 2, 4, 8 s

    # ---- Sandbox ----
    sandbox_mode: str = "docker"           # docker (mode A) | subprocess (mode B)
    docker_image: str = "eda-sandbox:1.1"   # 1.1 = built with requirements-lock.txt constraints
    docker_memory: str = "2g"
    docker_cpus: str = "2"
    docker_pids_limit: int = 256
    docker_tmpfs_size: str = "256m"
    step_timeout_s: float = 60.0           # soft timeout inside the worker (state kept)
    hard_timeout_grace_s: float = 10.0     # then the host kills the worker (state lost)
    sandbox_start_timeout_s: float = 60.0
    max_stdout_chars: int = 200_000        # cap kept by the worker per step
    # Excel sheet returned by load_data(path) without sheet= INSIDE the sandbox.
    # "first" = same as pd.read_excel: the loader gives the agent no sheet hint in
    # any eval configuration; only pre-profiling lists the sheets (GD6 ablation).
    sandbox_default_sheet: str = "first"

    # ---- Agent (GD4) ----
    max_steps: int = 10                    # LLM calls per task
    max_retries: int = 3                   # consecutive fix attempts after a failed execution
    max_format_errors: int = 2             # consecutive replies without code/FINAL tolerated
    reflexion: bool = True
    profiling: bool = True
    obs_max_chars: int = 2000
    profile_max_chars: int = 6000
    history_keep_steps: int = 4            # recent steps kept verbatim; older ones summarised

    # ---- Eval runner (GD5) ----
    eval_outage_wait_s: float = 60.0       # after an LLM outage: wait, then health-check
    eval_outage_checks: int = 5            # this many times before stopping the run (resume later)

    # ---- Paths ----
    datasets_dir: str = str(PROJECT_ROOT / "datasets")
    runs_dir: str = str(PROJECT_ROOT / "runs")

    def __post_init__(self) -> None:
        if self.sandbox_mode not in SANDBOX_MODES:
            raise ValueError(f"sandbox_mode must be one of {SANDBOX_MODES}, got {self.sandbox_mode!r}")
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ValueError(f"ollama_url must start with http:// or https://, got {self.ollama_url!r}")
        if self.sandbox_default_sheet not in ("first", "largest"):
            raise ValueError("sandbox_default_sheet must be 'first' or 'largest'")
        if self.max_retries < 0 or self.max_steps < 1:
            raise ValueError("max_retries must be >= 0 and max_steps >= 1")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def experiment_dict(self) -> dict:
        """Settings that can change results (infrastructure fields removed)."""
        return {k: v for k, v in self.to_dict().items() if k not in INFRA_FIELDS}

    def experiment_hash(self) -> str:
        """Same experiment = same hash, even with a new tunnel URL or another folder.
        eval/run_eval.py refuses to resume a run whose experiment hash differs."""
        blob = json.dumps(self.experiment_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def prompt_char_budget(self) -> int:
        """Max characters of the prompt (~3 chars/token, room left for the reply)."""
        return int((self.num_ctx - self.num_predict) * 3)

    def replace(self, **changes) -> "Config":
        return dataclasses.replace(self, **changes)


def _parse(raw: str, typ: type, name: str):
    if typ is bool:
        s = raw.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise ValueError(f"EDA_{name.upper()}: not a boolean: {raw!r}")
    if typ is int:
        return int(raw)
    if typ is float:
        return float(raw)
    return raw.strip()


def load_config(env: dict | None = None, **overrides) -> Config:
    """Defaults <- environment variables EDA_* <- explicit keyword overrides."""
    env = os.environ if env is None else env
    values = {}
    for f in fields(Config):
        key = f"EDA_{f.name.upper()}"
        if env.get(key, "") != "":
            typ = f.type if isinstance(f.type, type) else _TYPES[str(f.type)]
            values[f.name] = _parse(env[key], typ, f.name)
    values.update(overrides)
    return Config(**values)
