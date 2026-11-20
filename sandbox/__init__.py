"""Sandbox package: create_sandbox() picks the implementation from the config."""
from __future__ import annotations

from .base import ExecResult, Sandbox, SandboxError, format_observation, truncate_middle
from .docker_sandbox import DockerSandbox
from .subprocess_sandbox import SubprocessSandbox

__all__ = ["ExecResult", "Sandbox", "SandboxError", "DockerSandbox", "SubprocessSandbox",
           "create_sandbox", "format_observation", "truncate_middle"]


def create_sandbox(cfg, data_dir, output_dir) -> Sandbox:
    """cfg.sandbox_mode == "docker" (mode A) or "subprocess" (mode B)."""
    if cfg.sandbox_mode == "docker":
        return DockerSandbox(cfg, data_dir, output_dir)
    if cfg.sandbox_mode == "subprocess":
        return SubprocessSandbox(cfg, data_dir, output_dir)
    raise SandboxError(f"unknown sandbox_mode {cfg.sandbox_mode!r}")
