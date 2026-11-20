"""Pins: requirements files are ASCII, every direct pin equals the lock, the host
venv equals the lock, and the Docker image's pip freeze is a subset of the lock."""
import importlib.metadata as md
import subprocess
from pathlib import Path

import pytest

from config import load_config
from sandbox.docker_sandbox import docker_available, image_exists

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "requirements.txt", ROOT / "sandbox" / "requirements-sandbox.txt",
         ROOT / "requirements-lock.txt"]


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _pins(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-r")) and "==" in line:
            name, ver = line.split("==")
            out[_norm(name)] = ver.strip()
    return out


LOCK = _pins(ROOT / "requirements-lock.txt")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_requirement_files_are_ascii(path):
    path.read_bytes().decode("ascii")  # pip on Windows reads them with cp1252


def test_direct_pins_equal_lock():
    for path in FILES[:2]:
        for name, ver in _pins(path).items():
            assert LOCK.get(name) == ver, f"{path.name}: {name}=={ver} but lock has {LOCK.get(name)}"


def test_lock_covers_indirect_dependencies():
    for name in ("pydantic", "pydantic-core", "langsmith", "tzdata", "python-dateutil", "pytz"):
        assert name in LOCK, name


def test_host_venv_equals_lock():
    installed = {_norm(d.metadata["Name"]): d.version for d in md.distributions()}
    for name, ver in LOCK.items():
        assert installed.get(name) == ver, f"venv {name}={installed.get(name)} != lock {ver}"


CFG = load_config(env={})
_ok, _msg = docker_available()


@pytest.mark.docker
@pytest.mark.skipif(not (_ok and image_exists(CFG.docker_image)), reason=f"Docker/image not ready: {_msg}")
def test_image_freeze_is_subset_of_lock():
    out = subprocess.run(["docker", "run", "--rm", "--network", "none", CFG.docker_image, "pip", "freeze"],
                         capture_output=True, text=True, timeout=120).stdout
    image = {_norm(l.split("==")[0]): l.split("==")[1].strip() for l in out.splitlines() if "==" in l}
    assert len(image) >= 20
    for name, ver in image.items():
        assert LOCK.get(name) == ver, f"image {name}=={ver} but lock has {LOCK.get(name)}"
