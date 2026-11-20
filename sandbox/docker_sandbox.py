"""DockerSandbox (mode A): one long-lived, locked-down container per session.

The container's main process IS the worker (docker run -i ... worker.py), so:
  * variables persist across steps for the whole session;
  * a hard timeout / crash = `docker rm -f` + a fresh container (state_lost);
  * --rm and close() remove the container; every container carries the label
    eda-agent=sandbox so `python -m sandbox.docker_sandbox --cleanup` can remove
    leftovers after a hard crash of the host process.

Isolation flags: --network none, --memory/--memory-swap, --cpus, --pids-limit,
--read-only root FS + tmpfs /tmp, --cap-drop ALL, no-new-privileges,
non-root user 10001, data mounted read-only at /data, writable /output only.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path

from .base import DATA_PY_PATH, WORKER_PATH, Sandbox, SandboxError

LABEL = "eda-agent=sandbox"


def _docker(*args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except FileNotFoundError as e:
        raise SandboxError("docker CLI not found: install Docker Desktop and reopen the terminal") from e
    except subprocess.TimeoutExpired as e:
        raise SandboxError(f"docker {' '.join(args[:2])} timed out after {timeout:g} s; "
                           "is Docker Desktop running?") from e


def docker_available() -> tuple[bool, str]:
    try:
        r = _docker("info", "--format", "{{.ServerVersion}}", timeout=20)
    except SandboxError as e:
        return False, str(e)
    if r.returncode != 0:
        return False, ("Docker daemon not reachable (start Docker Desktop and wait for "
                       f"'Engine running'): {r.stderr.strip()[:300]}")
    return True, r.stdout.strip()


def image_exists(image: str) -> bool:
    return _docker("image", "inspect", image, timeout=30).returncode == 0


def _mount_src(p: Path) -> str:
    s = str(p.resolve())
    if "," in s:  # --mount uses commas as separators
        raise SandboxError(f"path contains a comma, cannot be mounted: {s}")
    return s


def cleanup_orphans() -> list[str]:
    """Remove ALL containers created by this project (use only when nothing is running)."""
    r = _docker("ps", "-aq", "--filter", f"label={LABEL}")
    ids = r.stdout.split()
    if ids:
        _docker("rm", "-f", *ids, timeout=120)
    return ids


class DockerSandbox(Sandbox):
    name = "docker"

    def __init__(self, cfg, data_dir, output_dir):
        super().__init__(cfg, data_dir, output_dir)
        self.container_name: str | None = None

    def _preflight(self) -> None:
        ok, msg = docker_available()
        if not ok:
            raise SandboxError(msg)
        if not image_exists(self.cfg.docker_image):
            raise SandboxError(f"Docker image {self.cfg.docker_image!r} not found. Build it with:\n"
                               f"  docker build -t {self.cfg.docker_image} -f sandbox/Dockerfile .")

    def _command(self):
        cfg = self.cfg
        self.container_name = f"eda-sbx-{uuid.uuid4().hex[:10]}"
        cmd = [
            "docker", "run", "--rm", "-i", "--init",
            "--name", self.container_name, "--label", LABEL,
            "--network", "none",
            "--memory", cfg.docker_memory, "--memory-swap", cfg.docker_memory,
            "--cpus", str(cfg.docker_cpus), "--pids-limit", str(cfg.docker_pids_limit),
            "--read-only", "--tmpfs", f"/tmp:rw,size={cfg.docker_tmpfs_size},mode=1777",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "10001:10001",
            "--mount", f"type=bind,source={_mount_src(self.data_dir)},target=/data,readonly",
            "--mount", f"type=bind,source={_mount_src(self.output_dir)},target=/output",
            "--mount", f"type=bind,source={_mount_src(WORKER_PATH)},target=/opt/eda/worker.py,readonly",
            "--mount", f"type=bind,source={_mount_src(DATA_PY_PATH)},target=/opt/eda/data.py,readonly",
            "-e", "EDA_DATA_DIR=/data", "-e", "EDA_OUTPUT_DIR=/output", "-e", "EDA_CODE_DIR=/opt/eda",
            "-e", f"EDA_MAX_STDOUT_CHARS={cfg.max_stdout_chars}",
            "-e", f"EDA_DEFAULT_SHEET={cfg.sandbox_default_sheet}",
            cfg.docker_image, "python", "-u", "/opt/eda/worker.py",
        ]
        return cmd, None, None

    def _kill_hard(self) -> None:
        if self.container_name:
            _docker("rm", "-f", self.container_name, timeout=60)
        if self.client:
            self.client.kill()

    def data_path(self, relative: str) -> str:
        return "/data/" + str(relative).replace("\\", "/").lstrip("/")

    def container_exists(self) -> bool:
        if not self.container_name:
            return False
        r = _docker("ps", "-aq", "--filter", f"name=^{self.container_name}$")
        return bool(r.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Docker sandbox utilities")
    ap.add_argument("--cleanup", action="store_true", help="remove leftover sandbox containers")
    ap.add_argument("--check", action="store_true", help="check daemon and image")
    args = ap.parse_args(argv)
    if args.cleanup:
        print(f"removed {len(cleanup_orphans())} container(s)")
    if args.check:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from config import load_config

        ok, msg = docker_available()
        print(f"docker daemon: {'OK ' + msg if ok else 'NOT OK: ' + msg}")
        img = load_config().docker_image
        print(f"image {img}: {'present' if ok and image_exists(img) else 'MISSING'}")
        return 0 if ok and image_exists(img) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
