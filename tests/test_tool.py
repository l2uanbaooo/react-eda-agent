"""GD3: the hand-written LangChain tool `python_repl` (@tool around OUR sandbox),
tested on the pinned langchain-core version, with both sandboxes."""
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from config import load_config
from sandbox import DockerSandbox, ExecResult, SubprocessSandbox
from sandbox.docker_sandbox import docker_available, image_exists
from sandbox.tool import make_python_repl_tool

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(env={}, step_timeout_s=4, hard_timeout_grace_s=8)
_ok, _msg = docker_available()
DOCKER_READY = _ok and image_exists(CFG.docker_image)


def _pinned(name: str) -> str:
    for line in (ROOT / "requirements.txt").read_text(encoding="ascii").splitlines():
        if line.startswith(name + "=="):
            return line.split("==")[1].strip()
    raise AssertionError(f"{name} not pinned")


def test_langchain_core_is_the_pinned_version():
    assert version("langchain-core") == _pinned("langchain-core")


@pytest.fixture(params=["subprocess", pytest.param("docker", marks=[
    pytest.mark.docker, pytest.mark.skipif(not DOCKER_READY, reason=f"Docker not ready: {_msg}")])])
def box(request, tmp_path):
    cls = SubprocessSandbox if request.param == "subprocess" else DockerSandbox
    b = cls(CFG, ROOT / "datasets", tmp_path).start()
    yield b
    b.close()


def test_tool_metadata(box):
    t = make_python_repl_tool(box)
    assert t.name == "python_repl"
    assert set(t.args) == {"code"}
    assert "sandbox" in t.description and "answer" in t.description


def test_tool_runs_code_in_the_sandbox(box):
    seen: list[ExecResult] = []
    t = make_python_repl_tool(box, on_result=seen.append)
    assert t.invoke({"code": "x = 40"}) == "(no output)"
    assert t.invoke({"code": "print(x + 2)"}) == "42"
    out = t.invoke({"code": "1/0"})
    assert "[error] ZeroDivisionError: division by zero" in out and "1/0" in out
    assert [r.ok for r in seen] == [True, True, False]
    where = t.invoke({"code": "import os; print(os.getuid() if hasattr(os, 'getuid') else 'windows')"})
    if isinstance(box, DockerSandbox):
        assert where == "10001"  # really executed inside the container


def test_tool_output_is_length_limited(box):
    t = make_python_repl_tool(box, max_chars=500)
    out = t.invoke({"code": "print('y' * 5000)"})
    assert len(out) < 600 and "characters truncated" in out


def test_no_host_executing_repl_tool_is_imported():
    import sandbox.tool  # noqa: F401
    assert "langchain_experimental" not in sys.modules
