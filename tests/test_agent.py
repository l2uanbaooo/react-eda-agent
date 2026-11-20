"""GD4: the ReAct agent with a SCRIPTED LLM (deterministic, no GPU) on real sandboxes.

Covers: action parsing, happy path scored by the real checker, self-debug and
recovery, retry limit (exactly 3 retries), reflexion on/off, format errors,
max steps, LLM outage, missing answer, answer reset, profiling on/off, history
trimming under the prompt budget, timeouts as incidents, plot task end-to-end,
JSONL log content, CLI, and a Docker run.
"""
import copy
import inspect
import json
import sys
from pathlib import Path

import pytest

import agent as A
from agent import JsonlLogger, ReActAgent, parse_action, run_spec, spec_from_task
from config import load_config
from llm import ChatResult, LLMUnavailable
from sandbox import DockerSandbox, SubprocessSandbox
from sandbox.docker_sandbox import docker_available, image_exists

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
GT = json.loads((DATASETS / "ground_truth.json").read_text(encoding="utf-8"))
TASKS = {t["id"]: t for t in json.loads((ROOT / "eval" / "tasks.json").read_text(encoding="utf-8"))["tasks"]}
sys.path.insert(0, str(ROOT / "eval"))
import checkers as C  # noqa: E402

BASE = load_config(env={}, sandbox_mode="subprocess", step_timeout_s=3, hard_timeout_grace_s=6)
SALES = "synthetic/sales.csv"


def py(code: str, thought: str = "Let me run this.") -> str:
    return f"{thought}\n```python\n{code}\n```"


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, num_predict=None):
        self.calls.append(copy.deepcopy(messages))
        if not self.replies:
            raise AssertionError("scripted LLM ran out of replies")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return ChatResult(text=r, model="scripted", prompt_tokens=11, completion_tokens=7,
                          duration_s=0.01, attempts=1, done_reason="stop")


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    b = SubprocessSandbox(BASE, DATASETS, tmp_path_factory.mktemp("agent_out")).start()
    yield b
    b.close()


def make_agent(box, replies, tmp_path=None, **cfg_over):
    cfg = BASE.replace(**cfg_over)
    llm = ScriptedLLM(replies)
    log = JsonlLogger(tmp_path / "log.jsonl" if tmp_path else None, run_id="t", task_id="t",
                      config_hash=cfg.config_hash(), sandbox=box.name, model="scripted")
    return ReActAgent(cfg, llm, box, log), llm, log


def run(box, replies, tmp_path=None, question="q", fmt="a number", extra="", profile=None, **over):
    ag, llm, log = make_agent(box, replies, tmp_path, **over)
    res = ag.run(question, box.data_path(SALES), fmt, extra, profile)
    log.close()
    return res, llm


def last_user(llm, call=-1) -> str:
    return llm.calls[call][-1]["content"]


LOAD = f"df = load_data(DATA_PATH).drop_duplicates()"


def load(box):
    return LOAD.replace("DATA_PATH", repr(box.data_path(SALES)))


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,kind,code,final", [
    ("Plan.\n```python\nx = 1\n```", "code", "x = 1", ""),
    ("```py\nprint(2)\n```", "code", "print(2)", ""),
    ("```\nprint(3)\n```", "code", "print(3)", ""),
    ("```python\na = 1\n```\nmore\n```python\nb = 2\n```", "code", "a = 1", ""),        # first block only
    ("```python\nx = 1\n```\nFINAL: done", "code", "x = 1", ""),                           # code wins
    ("All good.\nFINAL: total is 5.\nSecond line.", "final", "", "total is 5.\nSecond line."),
    ("**FINAL:** 42", "final", "", "42"),
    ("final: lowercase works", "final", "", "lowercase works"),
    ("```python\nx = 1\n", "invalid", "", ""),                                              # unclosed fence
    ("```python\n\n```", "invalid", "", ""),                                                # empty block
    ("I will now compute the mean.", "invalid", "", ""),
    ("", "invalid", "", ""),
])
def test_parse_action(text, kind, code, final):
    a = parse_action(text)
    assert (a.kind, a.code, a.final_text) == (kind, code, final)


# --------------------------------------------------------------------------
# Loop behaviour
# --------------------------------------------------------------------------
def test_happy_path_scored_by_real_checker(box, tmp_path):
    replies = [py(f"{load(box)}\nanswer = round(float(df['revenue'].sum()), 2)\nprint(answer)"),
               "FINAL: The total revenue after removing duplicates is printed above."]
    res, llm = run(box, replies, tmp_path)
    assert res.status == "final" and res.answer_found and res.n_exec == 1 and res.n_exec_errors == 0
    assert C.check(TASKS["G2_01"], res.answer, GT).passed
    assert res.llm_tokens == {"prompt": 22, "completion": 14}
    obs = last_user(llm)
    assert obs.startswith("OBSERVATION:") and str(res.answer) in obs


def test_self_debug_recovers_with_short_fix_prompt(box):
    bad = f"{load(box)}\nanswer = float(df['Revenue'].sum())"
    good = f"answer = round(float(df['revenue'].sum()), 2)\nprint(answer)"
    res, llm = run(box, [py(bad), py(good), "FINAL: done"])
    assert res.status == "final" and res.answer_found
    assert res.n_exec_errors == 1 and res.incidents[0]["type"] == "exec_error"
    assert res.incidents[0]["recovered"] is True
    fix = last_user(llm, 1)
    assert fix.startswith("The code failed (fix attempt 1 of 3).")
    assert "ERROR: KeyError: 'Revenue'" in fix
    assert "df['Revenue']" in fix                # failed code and traceback line shown
    assert "df.columns.tolist()" in fix          # hint for KeyError
    assert len(fix) < 2500                       # short


def test_observation_reports_answer_status(box):
    res, llm = run(box, [py("x = 2"), py("answer = 25\nprint(answer)"), "FINAL: 25"])
    assert "`answer` is not set yet." in last_user(llm, 1)
    assert "`answer` is currently set to: 25" in last_user(llm, 2)
    assert "reply with a line starting with FINAL:" in last_user(llm, 2)


def test_repeated_code_is_not_rerun_regression_g4_01(box, tmp_path):
    # Real Qwen run of G4_01: correct answer at step 1, then the SAME code 9 more times -> max_steps.
    code = (f"data = load_data({box.data_path(SALES)!r})\n"
            "duplicates = data.duplicated(keep='first')\nanswer = duplicates.sum()\nprint(answer)")
    replies = [py(code, "We use duplicated()."), py(code, ""), "FINAL: There are 25 duplicate rows."]
    res, llm = run(box, replies, tmp_path)
    assert res.status == "final" and res.answer == 25 and res.answer_found
    assert res.n_exec == 1 and res.n_repeated_code == 1      # the repeat was not executed
    nudge = last_user(llm, 2)
    assert nudge.startswith("You sent exactly the same code") and "currently set to: 25" in nudge
    assert C.check(TASKS["G4_01"], res.answer, GT).passed
    steps = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
             if json.loads(l)["event"] == "step"]
    assert steps[1].get("repeated_code") is True and "exec" not in steps[1]


def test_same_code_after_a_failure_is_still_executed(box):
    replies = [py("x = nope"), py("x = nope"), py("answer = 1"), "FINAL: 1"]
    res, _ = run(box, replies)
    assert res.n_exec == 3 and res.n_repeated_code == 0 and res.n_exec_errors == 2


def test_retry_limit_is_exactly_three(box):
    replies = [py("x = undefined_1"), py("x = undefined_2"), py("x = undefined_3"), py("x = undefined_4"),
               "FINAL: should never be reached"]
    res, llm = run(box, replies)
    assert res.status == "retry_exhausted"
    assert res.n_exec == 4 and res.n_exec_errors == 4 and res.max_consecutive_errors == 4
    assert len(llm.calls) == 4                   # 1 attempt + 3 retries, then stop
    assert [i["recovered"] for i in res.incidents] == [False] * 4
    assert "fix attempt 3 of 3" in last_user(llm, 3)


def test_errors_separated_by_success_do_not_exhaust_retries(box):
    replies = [py("a = nope1"), py("a = 1"), py("b = nope2"), py("b = 2"), py("c = nope3"),
               py("answer = a + b"), "FINAL: 3"]
    res, _ = run(box, replies)
    assert res.status == "final" and res.n_exec_errors == 3 and res.max_consecutive_errors == 1
    assert all(i["recovered"] for i in res.incidents)


@pytest.mark.parametrize("reflexion", [True, False])
def test_reflexion_lessons_only_when_enabled(box, reflexion):
    replies = [py("x = nope_a"), py("x = {}['k']"), py("x = 1/0"), py("answer = 1"), "FINAL: 1"]
    res, llm = run(box, replies, reflexion=reflexion)
    assert res.status == "final"
    first_fix, second_fix, third_fix = last_user(llm, 1), last_user(llm, 2), last_user(llm, 3)
    assert "LESSONS" not in first_fix                     # nothing to learn from yet
    if reflexion:
        assert "LESSONS FROM EARLIER ERRORS" in second_fix and "NameError" in second_fix
        assert "NameError" in third_fix and "KeyError" in third_fix
        assert len(res.lessons) == 3
    else:
        assert "LESSONS" not in second_fix and "LESSONS" not in third_fix and res.lessons == []


def test_format_errors_reminded_then_recovered(box):
    replies = ["Let me think about this.", "I will compute it.", py("answer = 5"), "FINAL: 5"]
    res, llm = run(box, replies)
    assert res.status == "final" and res.n_format_errors == 2 and res.answer == 5
    assert "no complete ```python code block" in last_user(llm, 1)


def test_too_many_format_errors_stop(box):
    res, llm = run(box, ["hmm", "hmm", "hmm", "never used"])
    assert res.status == "format_error" and len(llm.calls) == 3


def test_max_steps(box):
    res, llm = run(box, [py("x = 1")] * 3, max_steps=3)
    assert res.status == "max_steps" and res.steps == 3 and len(llm.calls) == 3


def test_llm_outage_is_reported(box, tmp_path):
    res, _ = run(box, [py("x = 1"), LLMUnavailable("tunnel down after 5 attempts")], tmp_path)
    assert res.status == "llm_error" and "tunnel down" in res.error
    events = [json.loads(l)["event"] for l in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "llm_error" in events and events[-1] == "end"


def test_final_without_answer_is_flagged(box):
    res, _ = run(box, [py("print(1 + 1)"), "FINAL: The answer is 2."])
    assert res.status == "final" and not res.answer_found and res.answer is None
    found, value, how = C.extract_answer_from_final(res.final_text, "number")
    assert found and value == 2.0 and how == "single_number"


def test_non_serializable_answer_is_flagged(box):
    res, _ = run(box, [py("answer = object()"), "FINAL: done"])
    assert not res.answer_found and "not JSON-serializable" in res.answer_error


def test_answer_from_previous_run_does_not_leak(box):
    box.run("answer = 'stale value'")
    res, _ = run(box, ["FINAL: nothing computed"])
    assert not res.answer_found and res.answer is None


def test_timeout_is_an_incident(box):
    res, llm = run(box, [py("while True:\n    pass"), py("answer = 1"), "FINAL: 1"], step_timeout_s=1)
    assert res.incidents[0]["type"] == "timeout" and res.incidents[0]["recovered"]
    assert "took too long" in last_user(llm, 1)


@pytest.mark.parametrize("profiling", [True, False])
def test_profile_only_when_profiling_enabled(box, profiling):
    from data import profile_file
    prof = profile_file(DATASETS / "synthetic/hr.xlsx", max_chars=3000)
    res, llm = run(box, ["FINAL: x"], profile=prof, profiling=profiling)
    task_msg = llm.calls[0][1]["content"]
    assert ("DATA PROFILE" in task_msg) is profiling
    assert ("SHEETS: README" in task_msg) is profiling   # the only sheet hint the agent can get
    system = llm.calls[0][0]["content"]
    assert "sheet=None returns the FIRST sheet" in system


def test_history_is_trimmed_under_budget(box):
    noisy = [py(f"v{i} = {i}\nprint('x' * 1900)") for i in range(8)] + [py("answer = v7"), "FINAL: 7"]
    res, llm = run(box, noisy, num_ctx=4096, num_predict=1024, max_steps=12)
    assert res.status == "final" and res.answer == 7
    budget = BASE.replace(num_ctx=4096, num_predict=1024).prompt_char_budget()
    for call in llm.calls:
        assert sum(len(m["content"]) for m in call) <= budget
        assert call[0]["role"] == "system" and call[1]["content"].startswith("TASK: q")
    assert "EARLIER STEPS (summarised" in llm.calls[-1][1]["content"]
    assert "step 1: ran `v0 = 0" in llm.calls[-1][1]["content"]


def test_log_lines_are_complete(box, tmp_path):
    replies = [py("x = nope"), py("answer = 1"), "FINAL: 1"]
    run(box, replies, tmp_path)
    recs = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in recs] == ["start", "step", "step", "step", "end"]
    for r in recs:
        assert {"ts", "run_id", "task_id", "config_hash", "sandbox", "model", "event"} <= set(r)
    start, s1, s2, s3, end = recs
    assert start["config"]["max_retries"] == 3 and start["versions"]["pandas"] == "2.2.3"
    assert s1["code"] == "x = nope" and s1["exec"]["ok"] is False and s1["retry_index"] == 1
    assert s1["exec"]["error"].startswith("NameError") and "thought" in s1 and "llm" in s1
    assert s2["exec"]["ok"] is True and s2["retry_index"] == 0
    assert s3["kind"] == "final" and s3["answer"] == 1
    assert end["status"] == "final" and end["incidents"][0]["recovered"] is True


def test_execution_goes_through_the_langchain_tool(box, monkeypatch):
    ag, llm, _ = make_agent(box, [py("answer = 3"), "FINAL: 3"])
    calls = []
    original = ag.tool.invoke
    monkeypatch.setattr(type(ag.tool), "invoke", lambda self, inp, *a, **k: calls.append(inp) or original(inp))
    res = ag.run("q", box.data_path(SALES), "a number")
    assert ag.tool.name == "python_repl" and calls == [{"code": "answer = 3"}] and res.answer == 3


def test_react_loop_stays_small():
    src = inspect.getsource(ReActAgent)
    assert src.count("\n") <= 200, "the hand-written ReAct loop should stay ~150-200 lines"


# --------------------------------------------------------------------------
# Task specs, plot task end-to-end, CLI
# --------------------------------------------------------------------------
def test_spec_from_task_adds_save_fig_for_plots():
    s = spec_from_task(TASKS["G5_01"])
    assert "save_fig(fig, 'monthly_revenue')" in s.extra and "answer = 'monthly_revenue'" in s.extra
    assert spec_from_task(TASKS["G2_01"]).extra == ""


def test_plot_task_end_to_end_with_run_spec(tmp_path):
    cfg = BASE.replace(runs_dir=str(tmp_path))
    code = (f"df = load_data({str(DATASETS / SALES)!r}).drop_duplicates()\n"
            "m = df.groupby(df['order_date'].str[:7])['revenue'].sum().reset_index()\n"
            "fig = px.line(m, x='order_date', y='revenue')\n"
            "save_fig(fig, 'monthly_revenue')\nanswer = 'monthly_revenue'")
    llm = ScriptedLLM([py(code), "FINAL: monthly revenue chart saved."])
    res, log_path, out_dir = run_spec(cfg, llm, spec_from_task(TASKS["G5_01"]), tmp_path / "run1")
    assert res.status == "final" and log_path.exists()
    assert "save_fig(fig, 'monthly_revenue')" in llm.calls[0][1]["content"]   # instruction reached the model
    assert "DATA PROFILE" in llm.calls[0][1]["content"]
    chk = C.check(TASKS["G5_01"], res.answer, GT, artifacts_dir=out_dir)
    assert chk.passed, chk.reason


def test_cli_runs_a_task_and_checks_it(tmp_path, monkeypatch, capsys):
    code = (f"df = load_data({str(DATASETS / SALES)!r})\n"
            "answer = int(df.duplicated().sum())\nprint(answer)")
    llm = ScriptedLLM([py(code), "FINAL: 25 duplicate rows."])
    monkeypatch.setenv("EDA_SANDBOX_MODE", "subprocess")
    monkeypatch.setenv("EDA_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(A, "_make_llm", lambda cfg: llm)
    assert A.main(["--task", "G4_01", "--run-id", "cli"]) == 0
    out = capsys.readouterr().out
    assert "status=final" in out and "CHECK G4_01: PASS" in out
    assert (tmp_path / "cli" / "logs" / "G4_01.jsonl").exists()


# --------------------------------------------------------------------------
# Docker
# --------------------------------------------------------------------------
_ok, _msg = docker_available()


@pytest.mark.docker
@pytest.mark.skipif(not (_ok and image_exists(BASE.docker_image)), reason=f"Docker not ready: {_msg}")
def test_agent_on_docker_with_self_debug(tmp_path):
    cfg = BASE.replace(sandbox_mode="docker")
    with DockerSandbox(cfg, DATASETS, tmp_path) as sb:
        path = sb.data_path("synthetic/eu_sales.csv")
        replies = [py(f"df = pd.read_csv({path!r})\nanswer = float(df['revenue_eur'].sum())"),  # naive read fails
                   py(f"df = load_data({path!r})\n"
                      "answer = round(float((df.revenue_eur.sum() - df.cost_eur.sum()) / df.revenue_eur.sum() * 100), 2)\n"
                      "print(answer)"),
                   "FINAL: gross margin computed."]
        ag = ReActAgent(cfg, ScriptedLLM(replies), sb)
        res = ag.run("gross margin", path, "a number")
    assert res.status == "final" and res.sandbox == "docker"
    assert res.n_exec_errors == 1 and res.incidents[0]["recovered"]
    assert C.check(TASKS["G2_05"], res.answer, GT).passed
