"""GD3: sandbox tests.

Part 1 runs the SAME behaviour tests on both implementations (subprocess and
Docker): state kept across steps, errors with the failing source line,
load_data inside the sandbox, save_fig + plot checker end-to-end, `answer`.
Part 2 is Docker-only and runs on the REAL Docker daemon: no network,
read-only data and root FS, non-root user, soft timeout keeps state, hard
timeout kills and restarts, memory and pids limits, pinned versions, cleanup.

Run: .venv\\Scripts\\python.exe -m pytest tests\\test_sandbox.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from config import load_config
from sandbox import DockerSandbox, ExecResult, SubprocessSandbox, format_observation, truncate_middle
from sandbox.docker_sandbox import docker_available, image_exists

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
GT = json.loads((DATASETS / "ground_truth.json").read_text(encoding="utf-8"))
TASKS = {t["id"]: t for t in json.loads((ROOT / "eval" / "tasks.json").read_text(encoding="utf-8"))["tasks"]}
sys.path.insert(0, str(ROOT / "eval"))
import checkers as C  # noqa: E402

CFG = load_config(env={}, step_timeout_s=4, hard_timeout_grace_s=8)
_ok, _msg = docker_available()
DOCKER_READY = _ok and image_exists(CFG.docker_image)
needs_docker = pytest.mark.skipif(not DOCKER_READY,
                                  reason=f"Docker not ready ({_msg}) or image {CFG.docker_image} missing")


@pytest.fixture(scope="module", params=["subprocess", pytest.param("docker", marks=[pytest.mark.docker,
                                                                                     needs_docker])])
def sb(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"out_{request.param}")
    cls = SubprocessSandbox if request.param == "subprocess" else DockerSandbox
    box = cls(CFG, DATASETS, out).start()
    yield box
    box.close()


# --------------------------------------------------------------------------
# Part 1: behaviour shared by both sandboxes
# --------------------------------------------------------------------------
def test_state_persists_across_steps(sb):
    assert sb.run("x = 41").ok
    r = sb.run("print(x + 1)")
    assert r.ok and r.stdout == "42\n"


def test_error_reports_type_and_failing_line(sb):
    r = sb.run("a = 1\nb = undefined_name + a\n")
    assert not r.ok and r.error == "NameError: name 'undefined_name' is not defined"
    assert "b = undefined_name + a" in r.traceback  # source line visible for self-debugging
    assert "worker.py" not in r.traceback
    assert sb.run("print('still alive')").stdout == "still alive\n"


def test_syntax_error_and_exit_are_contained(sb):
    r = sb.run("def f(:\n  pass")
    assert not r.ok and r.error.startswith("SyntaxError")
    r = sb.run("import sys; sys.exit(3)")
    assert not r.ok and "not allowed" in r.error
    assert sb.run("print(1)").ok and sb.restarts == 0


def test_unicode_output(sb):
    r = sb.run("print('Hà Nội, Đà Nẵng ✓')")
    assert r.ok and r.stdout == "Hà Nội, Đà Nẵng ✓\n"


def test_stderr_and_warnings_captured(sb):
    r = sb.run("import sys, warnings\nprint('to err', file=sys.stderr)\nwarnings.warn('careful')")
    assert r.ok and "to err" in r.stderr and "careful" in r.stderr


@pytest.mark.parametrize("ds", ["sales", "hr", "sensors", "customers", "eu_sales", "clinic", "transactions"])
def test_load_data_inside_sandbox(sb, ds):
    rel = GT["datasets"][ds]["file"]
    sheet = ", sheet='employees'" if ds == "hr" else ""  # sandbox default is the FIRST sheet (README)
    r = sb.run(f"df = load_data({sb.data_path(rel)!r}{sheet})\nprint(len(df), df.attrs['source']['format'])")
    assert r.ok, r.error
    facts = GT["datasets"][ds]["facts"]
    expected_rows = facts.get("n_rows_raw") or facts.get("n_rows") or facts.get("n_employees")
    assert int(r.stdout.split()[0]) == expected_rows


def test_sandbox_load_data_gives_no_sheet_hint(sb):
    # Inside the sandbox, load_data(path) on a workbook returns the FIRST sheet,
    # like pd.read_excel, and no list of sheets (only pre-profiling lists them).
    r = sb.run(f"df = load_data({sb.data_path('synthetic/hr.xlsx')!r})\n"
               "print(df.attrs['source'], list(df.columns))")
    assert r.ok, r.error
    assert "'sheet': 'README'" in r.stdout and "'sheets'" not in r.stdout and "['note']" in r.stdout


def test_answer_variable(sb):
    sb.run("answer = None")
    assert sb.get_answer() == (False, None, None)
    sb.run("answer = {'n': np.int64(25), 'r': np.float64(0.5), 's': pd.Series({'a': 1}), 'nan': float('nan')}")
    assert sb.get_answer() == (True, {"n": 25, "r": 0.5, "s": {"a": 1}, "nan": None}, None)
    sb.run("answer = lambda: 1")
    found, value, err = sb.get_answer()
    assert found and value is None and "not JSON-serializable" in err
    sb.run("answer = None")


def test_save_fig_end_to_end_with_plot_checker(sb):
    code = f"""
df = load_data({sb.data_path('synthetic/sales.csv')!r}).drop_duplicates()
m = df.groupby(pd.to_datetime(df['order_date']).dt.strftime('%Y-%m'))['revenue'].sum().reset_index()
m.columns = ['month', 'revenue']
fig = px.line(m, x='month', y='revenue', title='Monthly revenue')
save_fig(fig, 'monthly_revenue')
answer = 'monthly_revenue'
"""
    r = sb.run(code)
    assert r.ok, r.error
    assert "[save_fig] saved figure 'monthly_revenue'" in r.stdout
    assert {"monthly_revenue.json", "monthly_revenue.html"} <= set(sb.list_outputs())
    found, value, _ = sb.get_answer()
    res = C.check(TASKS["G5_01"], value, GT, artifacts_dir=sb.output_dir)
    assert res.passed, res.reason


def test_save_fig_rejects_bad_input(sb):
    r = sb.run("save_fig(fig, '../escape')")
    assert not r.ok and "must match" in r.error
    r = sb.run("save_fig({'not': 'a figure'}, 'x')")
    assert not r.ok and "expects a Plotly figure" in r.error


def test_large_output_is_capped_and_truncated(sb):
    r = sb.run("print('x' * 300000)")
    assert r.ok and len(r.stdout) < 210_000 and "cut by sandbox" in r.stdout
    obs = format_observation(r, max_chars=2000)
    assert len(obs) < 2100 and "characters truncated" in obs


def test_timeout_kills_runaway_code(sb):
    sb.run("keep_me = 7")
    r = sb.run("while True:\n    pass", timeout=2)
    assert not r.ok and r.timed_out
    if sb.ready_info.get("alarm"):  # Linux worker (Docker, Kaggle): soft timeout keeps state
        assert not r.state_lost and sb.run("print(keep_me)").stdout == "7\n"
    else:                            # Windows subprocess: only the hard kill exists
        assert r.state_lost and "restarted" in r.error
        assert "NameError" in sb.run("print(keep_me)").error


# --------------------------------------------------------------------------
# Part 2: Docker isolation on the real daemon
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dsb(tmp_path_factory):
    if not DOCKER_READY:
        pytest.skip(f"Docker not ready: {_msg}")
    box = DockerSandbox(CFG, DATASETS, tmp_path_factory.mktemp("docker_iso")).start()
    yield box
    box.close()


@pytest.mark.docker
@needs_docker
def test_no_network(dsb):
    r = dsb.run("import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)")
    assert not r.ok and "Network is unreachable" in r.error
    r = dsb.run("import socket\nsocket.getaddrinfo('example.com', 443)")
    assert not r.ok and "gaierror" in r.error


@pytest.mark.docker
@needs_docker
def test_data_is_read_only(dsb):
    r = dsb.run("open('/data/synthetic/sales.csv', 'a').write('x')")
    assert not r.ok and "Read-only file system" in r.error
    r = dsb.run("import os; os.remove('/data/ground_truth.json')")
    assert not r.ok and "Read-only file system" in r.error
    assert dsb.run("import os; print(sorted(os.listdir('/data')))").ok  # reading still works


@pytest.mark.docker
@needs_docker
def test_root_fs_read_only_tmp_and_output_writable(dsb):
    r = dsb.run("open('/etc/evil', 'w').write('x')")
    assert not r.ok and ("Read-only file system" in r.error or "Permission denied" in r.error)
    r = dsb.run("open('/tmp/t.txt', 'w').write('ok'); open('/output/o.txt', 'w').write('ok'); print('ok')")
    assert r.ok, r.error
    assert (dsb.output_dir / "o.txt").read_text() == "ok"


@pytest.mark.docker
@needs_docker
def test_runs_as_non_root(dsb):
    r = dsb.run("import os; print(os.getuid(), os.getgid())")
    assert r.stdout.split() == ["10001", "10001"]
    assert dsb.ready_info["uid"] == 10001


@pytest.mark.docker
@needs_docker
def test_container_flags(dsb):
    out = subprocess.run(["docker", "inspect", dsb.container_name], capture_output=True, text=True)
    info = json.loads(out.stdout)[0]
    hc = info["HostConfig"]
    assert hc["NetworkMode"] == "none" and hc["ReadonlyRootfs"] is True
    assert hc["Memory"] == 2 * 1024 ** 3 and hc["PidsLimit"] == CFG.docker_pids_limit
    assert hc["CapDrop"] == ["ALL"] and "no-new-privileges" in hc["SecurityOpt"]
    mounts = {m["Destination"]: m for m in info["Mounts"]}
    assert mounts["/data"]["RW"] is False and mounts["/output"]["RW"] is True
    assert mounts["/opt/eda/worker.py"]["RW"] is False and mounts["/opt/eda/data.py"]["RW"] is False
    assert info["Config"]["Labels"].get("eda-agent") == "sandbox"


@pytest.mark.docker
@needs_docker
def test_soft_timeout_keeps_state_in_docker(dsb):
    dsb.run("big = list(range(5))")
    r = dsb.run("import time\nwhile True:\n    time.sleep(0.1)", timeout=2)
    assert r.timed_out and not r.state_lost and "interrupted" in r.error
    assert dsb.run("print(len(big))").stdout == "5\n"


@pytest.mark.docker
@needs_docker
def test_soft_timeout_cannot_be_swallowed_by_except_exception(dsb):
    code = "try:\n    while True:\n        pass\nexcept Exception:\n    print('swallowed')"
    r = dsb.run(code, timeout=2)
    assert r.timed_out and "swallowed" not in r.stdout


@pytest.mark.docker
@needs_docker
def test_hard_timeout_kills_container_and_restarts(tmp_path):
    box = DockerSandbox(CFG, DATASETS, tmp_path).start()
    try:
        first = box.container_name
        box.run("x = 1")
        code = "import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nwhile True:\n    pass"
        r = box.run(code, timeout=2)
        assert r.timed_out and r.state_lost and "restarted" in r.error
        assert box.restarts == 1 and box.container_name != first
        assert "NameError" in box.run("print(x)").error   # fresh session
        assert box.run("print(1 + 1)").stdout == "2\n"
        gone = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=^{first}$"],
                              capture_output=True, text=True).stdout.strip()
        assert gone == ""
    finally:
        box.close()


@pytest.mark.docker
@needs_docker
def test_memory_limit_kills_and_reports(tmp_path):
    box = DockerSandbox(CFG.replace(docker_memory="256m"), DATASETS, tmp_path).start()
    try:
        box.run("y = 1")
        r = box.run("block = bytearray(700 * 1024 * 1024)")
        assert not r.ok and r.state_lost
        assert ("SandboxCrash" in r.error) or ("MemoryError" in r.error)
        assert box.run("print('alive')").stdout == "alive\n"
    finally:
        box.close()


@pytest.mark.docker
@needs_docker
def test_pids_limit(tmp_path):
    box = DockerSandbox(CFG.replace(docker_pids_limit=64), DATASETS, tmp_path).start()
    try:
        code = ("import threading, time\nts = []\n"
                "for i in range(200):\n"
                "    t = threading.Thread(target=time.sleep, args=(5,)); t.start(); ts.append(t)\n")
        r = box.run(code, timeout=20)
        assert not r.ok and "can't start new thread" in r.error
    finally:
        box.close()


@pytest.mark.docker
@needs_docker
def test_image_versions_match_pins_and_host(dsb):
    import importlib.metadata as md

    pins = {}
    for line in (ROOT / "sandbox" / "requirements-sandbox.txt").read_text(encoding="ascii").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, ver = line.split("==")
            pins[name] = ver
    image = dsb.versions()
    assert image["python"].startswith("3.11")
    for name, ver in pins.items():
        assert image[name] == ver, f"image {name}={image[name]} != pin {ver}"
        assert md.version(name) == ver, f"host venv {name}={md.version(name)} != pin {ver}"


@pytest.mark.docker
@needs_docker
def test_close_removes_container(tmp_path):
    box = DockerSandbox(CFG, DATASETS, tmp_path).start()
    assert box.container_exists()
    box.close()
    assert not box.container_exists()


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def test_format_observation_and_truncation():
    r = ExecResult(ok=False, stdout="partial\n", error="KeyError: 'x'",
                   traceback="Traceback...\n  File \"<step3>\", line 1\n    df['x']\nKeyError: 'x'",
                   state_lost=True)
    obs = format_observation(r)
    assert "partial" in obs and "[error] KeyError: 'x'" in obs and "df['x']" in obs
    assert "all variables were lost" in obs
    assert format_observation(ExecResult(ok=True)) == "(no output)"
    t = truncate_middle("a" * 100 + "b" * 100, 50)
    assert t.startswith("a" * 30) and t.endswith("b" * 20) and "150 characters truncated" in t
