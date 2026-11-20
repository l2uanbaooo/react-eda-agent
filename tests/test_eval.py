"""GD5: eval runner (resume, manifest, outages, answer fallback) and score metrics.

The runner is exercised end-to-end with a routed SCRIPTED LLM on the real
subprocess sandbox and the real checkers; score.py is tested on hand-made
result files with known numbers, and its lint on real pylint/ruff runs.
"""
import json
import sys
from pathlib import Path

import pytest

from config import load_config
from llm import ChatResult, LLMUnavailable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import run_eval as R  # noqa: E402
import score as S  # noqa: E402

TASKS = {t["id"]: t for t in R.load_tasks()}
SALES = str(ROOT / "datasets" / "synthetic" / "sales.csv")


def py(code):
    return f"Running.\n```python\n{code}\n```"


class RoutedLLM:
    """Scripted replies per task, chosen from the TASK line of the prompt."""

    def __init__(self, scripts: dict, healthy=True):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.calls: list[str] = []
        self.healthy = list(healthy) if isinstance(healthy, list) else healthy

    def _task_id(self, messages):
        text = messages[1]["content"]
        for tid, t in TASKS.items():
            if text.startswith(f"TASK: {t['question']}"):
                return tid
        raise AssertionError("unknown task prompt")

    def chat(self, messages, num_predict=None):
        tid = self._task_id(messages)
        self.calls.append(tid)
        if not self.scripts.get(tid):
            raise AssertionError(f"no scripted reply left for {tid}")
        r = self.scripts[tid].pop(0)
        if isinstance(r, BaseException):
            raise r
        return ChatResult(r, "scripted", 5, 5, 0.0, 1, "stop")

    def health(self, warmup=True):
        ok = self.healthy.pop(0) if isinstance(self.healthy, list) else self.healthy
        return {"ok": ok, "error": None if ok else "down"}


def cfg_for(tmp_path, **over):
    return load_config(env={}, sandbox_mode="subprocess", step_timeout_s=5, hard_timeout_grace_s=8,
                       runs_dir=str(tmp_path), eval_outage_wait_s=0, eval_outage_checks=2, **over)


SCRIPTS = {
    "G4_01": [py(f"df = load_data({SALES!r})\nanswer = int(df.duplicated().sum())\nprint(answer)"),
              "FINAL: 25 duplicate rows."],
    "G2_01": [py(f"df = load_data({SALES!r}).drop_duplicates()\nprint(round(df['revenue'].sum(), 2))"),
              "FINAL: The total revenue is 1107048.97."],                # answer NOT set -> fallback
    "G1_02": [py("answer = {'sheet': 'README', 'n_employees': 5}"), "FINAL: README"],   # wrong
    "G3_01": [py("x = nope"), py("x = nope2"), py("x = nope3"), py("x = nope4")],     # retry exhausted
}


def run(tmp_path, ids, scripts=None, run_id="r1", **over):
    llm = RoutedLLM(scripts if scripts is not None else SCRIPTS)
    logs = []
    out = R.run_eval(cfg_for(tmp_path, **over), [TASKS[i] for i in ids], run_id, llm=llm,
                     sleep=lambda s: None, log=logs.append)
    return out, llm, logs


def result(tmp_path, tid, run_id="r1"):
    return json.loads((tmp_path / run_id / "results" / f"{tid}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def test_run_writes_results_and_manifest(tmp_path):
    out, llm, logs = run(tmp_path, ["G4_01", "G2_01", "G1_02", "G3_01"])
    assert out["exit_code"] == 0 and out["ran"] == 4
    man = json.loads((tmp_path / "r1" / "manifest.json").read_text(encoding="utf-8"))
    assert man["sandbox"] == "subprocess" and man["model"] == "qwen2.5-coder:14b"
    assert man["tasks"] == ["G1_02", "G2_01", "G3_01", "G4_01"] and len(man["experiment_hash"]) == 12

    g4 = result(tmp_path, "G4_01")
    assert g4["passed"] and g4["answer_source"] == "answer" and g4["error_codes"] == {}
    assert (tmp_path / "r1" / g4["log"]).exists() and g4["successful_code"]

    g2 = result(tmp_path, "G2_01")                       # fallback from FINAL counts, but is flagged
    assert g2["passed"] and g2["answer_source"] == "single_number"
    assert g2["error_codes"] == {"answer_missing": 1}

    g1 = result(tmp_path, "G1_02")
    assert not g1["passed"] and g1["check"]["status"] == "FAIL"

    g3 = result(tmp_path, "G3_01")
    assert g3["status"] == "retry_exhausted" and not g3["passed"] and g3["answer_source"] == "none"
    assert g3["error_codes"] == {"exec_error": 4, "retry_exhausted": 1}
    assert any("-> PASS" in line for line in logs)


def test_resume_skips_finished_tasks(tmp_path):
    run(tmp_path, ["G4_01"])
    out, llm, logs = run(tmp_path, ["G4_01", "G1_02"], scripts={"G1_02": SCRIPTS["G1_02"]})
    assert out["ran"] == 1 and set(llm.calls) == {"G1_02"}   # G4_01 not asked again
    assert "1 already done" in logs[0]


def test_interrupted_run_resumes(tmp_path):
    scripts = {"G4_01": SCRIPTS["G4_01"], "G1_02": [KeyboardInterrupt()]}
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, ["G4_01", "G1_02"], scripts=scripts)
    assert result(tmp_path, "G4_01")["passed"]
    assert not (tmp_path / "r1" / "results" / "G1_02.json").exists()
    out, llm, _ = run(tmp_path, ["G4_01", "G1_02"], scripts={"G1_02": SCRIPTS["G1_02"]})
    assert out["ran"] == 1 and llm.calls == ["G1_02", "G1_02"]


def test_llm_outage_stops_then_resume_reruns_only_that_task(tmp_path):
    scripts = {"G4_01": [LLMUnavailable("tunnel down")]}
    llm = RoutedLLM(scripts, healthy=[True, False, False])     # start ok, then 2 failed checks
    logs = []
    out = R.run_eval(cfg_for(tmp_path), [TASKS["G4_01"], TASKS["G1_02"]], "r1", llm=llm,
                     sleep=lambda s: None, log=logs.append)
    assert out["exit_code"] == 4 and out["stopped"] == "llm_outage"
    infra = result(tmp_path, "G4_01")
    assert infra["infra_error"] and infra["check"] is None and infra["status"] == "llm_error"
    out2, llm2, _ = run(tmp_path, ["G4_01", "G1_02"])         # resume with a working LLM
    assert out2["exit_code"] == 0 and out2["ran"] == 2
    assert result(tmp_path, "G4_01")["passed"] and not result(tmp_path, "G4_01")["infra_error"]


def test_llm_outage_recovers_and_retries_the_task(tmp_path):
    scripts = {"G4_01": [LLMUnavailable("blip")] + SCRIPTS["G4_01"]}
    llm = RoutedLLM(scripts, healthy=[True, False, True])
    out = R.run_eval(cfg_for(tmp_path), [TASKS["G4_01"]], "r1", llm=llm, sleep=lambda s: None, log=lambda m: None)
    assert out["exit_code"] == 0 and result(tmp_path, "G4_01")["passed"]


def test_llm_not_ready_at_start(tmp_path):
    llm = RoutedLLM({}, healthy=False)
    out = R.run_eval(cfg_for(tmp_path), [TASKS["G4_01"]], "r1", llm=llm, log=lambda m: None)
    assert out["exit_code"] == 3 and not (tmp_path / "r1" / "results").exists()


def test_manifest_refuses_a_different_experiment(tmp_path):
    run(tmp_path, ["G4_01"])
    out, _, logs = run(tmp_path, ["G1_02"], reflexion=False)
    assert out["exit_code"] == 6 and "reflexion" in logs[0]


def test_new_tunnel_url_does_not_change_the_experiment(tmp_path):
    a = cfg_for(tmp_path)
    b = a.replace(ollama_url="https://new-url.trycloudflare.com", runs_dir="elsewhere")
    assert a.experiment_hash() == b.experiment_hash() and a.config_hash() != b.config_hash()
    assert a.replace(profiling=False).experiment_hash() != a.experiment_hash()
    assert a.replace(sandbox_mode="docker").experiment_hash() != a.experiment_hash()


def test_select_tasks():
    all_tasks = R.load_tasks()
    assert [t["id"] for t in R.select_tasks(all_tasks, "G1_0[12],G5_06")] == ["G1_01", "G1_02", "G5_06"]
    assert len(R.select_tasks(all_tasks, subset="chatgpt")) == 12
    assert len(R.select_tasks(all_tasks, "G6_*", "chatgpt")) == 2


def test_cli_main_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EDA_SANDBOX_MODE", "subprocess")
    monkeypatch.setenv("EDA_RUNS_DIR", str(tmp_path))
    llm = RoutedLLM({"G4_01": SCRIPTS["G4_01"]})
    monkeypatch.setattr(R, "OllamaClient", lambda cfg: llm)
    assert R.main(["--run-id", "cli", "--tasks", "G4_01", "--score"]) == 0
    out = capsys.readouterr().out
    assert "resume/continue with" in out and "| K1 task completion | 100.0% (1/1" in out
    assert (tmp_path / "cli" / "summary.json").exists()


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------
def fake(tid, group, passed, *, infra=False, source="answer", incidents=(), errs=0, steps=3,
         codes=None, code=("answer = 1",)):
    return {"task_id": tid, "group": group, "passed": passed, "infra_error": infra,
            "status": "llm_error" if infra else "final", "answer_source": source,
            "check": None if infra else {"reason": "ok" if passed else "wrong"},
            "error_codes": codes or {}, "incidents": list(incidents), "n_exec_errors": errs,
            "n_exec": steps - 1, "steps": steps, "n_repeated_code": 0, "duration_s": 10.0,
            "llm_tokens": {"prompt": 100, "completion": 10}, "sandbox": "docker", "model": "m",
            "experiment_hash": "abc", "successful_code": list(code)}


def inc(rec, typ="exec_error"):
    return {"exec": 1, "step": 1, "type": typ, "error": "E", "recovered": rec}


def test_wilson_interval():
    assert S.wilson(0, 0) == (None, None)
    lo, hi = S.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    lo, hi = S.wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)


def test_summary_numbers():
    results = [
        fake("G1_01", "G1", True),
        fake("G1_02", "G1", True, source="single_number", codes={"answer_missing": 1}),
        fake("G2_01", "G2", False, incidents=[inc(True), inc(False)], errs=2, codes={"exec_error": 2}),
        fake("G2_02", "G2", True, incidents=[inc(True, "timeout")], errs=1, codes={"timeout": 1}),
        fake("G3_01", "G3", False, infra=True),
        fake("G3_02", "G3", False, errs=4, incidents=[inc(False)] * 4, codes={"exec_error": 4}),
    ]
    s = S.summarize(results)
    assert s["n_scored"] == 5 and s["infra_error_tasks"] == ["G3_01"]   # infra excluded
    assert s["K1_task_completion"]["k"] == 3 and s["K1_task_completion"]["n"] == 5
    assert s["K1_task_completion"]["rate"] == 0.6 and s["targets"]["K1>=0.80"] is False
    assert s["K1_strict_answer_variable_only"]["k"] == 2
    assert s["K1_by_group"]["G1"]["rate"] == 1.0 and s["K1_by_group"]["G3"]["n"] == 1
    assert s["K1_by_group"]["G5"]["rate"] is None
    k2 = s["K2_self_debug"]
    assert (k2["k"], k2["n"]) == (2, 7) and k2["small_sample"] is True
    assert s["K2_strict_recovered_and_task_passed"]["k"] == 1            # only G2_02's incident
    assert s["K2_by_type"]["timeout"]["k"] == 1
    assert s["E2_retries"]["distribution"] == {"0": 2, "1": 1, "2": 1, "3+": 1}
    assert s["error_codes"]["exec_error"] == {"tasks": 2, "occurrences": 6}
    assert s["error_codes"]["answer_missing"]["tasks"] == 1


def test_summary_markdown_is_honest():
    s = S.summarize([fake("G1_01", "G1", True)])
    md = S.to_markdown("x", s, None)
    assert "100.0% (1/1; 95% CI" in md and "SMALL SAMPLE" in md and "MET" in md
    assert "| K2 self-debug success | n/a (no data) | >= 70%: n/a |" in md
    s2 = S.summarize([fake("G1_01", "G1", False)])
    assert "NOT MET" in S.to_markdown("x", s2, None)
    mixed = S.summarize([fake("a", "G1", True), {**fake("b", "G1", True), "sandbox": "subprocess"}])
    assert "MIXED" in S.to_markdown("x", mixed, None)


def test_lint_with_real_pylint_and_ruff(tmp_path):
    results = [fake("G1_01", "G1", True, code=("df = load_data(DATA_DIR + '/s.csv')",
                                               "answer = int(df.shape[0])\nprint(answer)")),
               fake("G1_02", "G1", True, code=("import os\nanswer = undefined_name",)),
               fake("G1_03", "G1", False, code=())]
    lint = S.lint_run(results, tmp_path)
    assert lint["G1_01"]["pylint_score"] == 10.0 and lint["G1_01"]["ruff_F"] == 0  # sandbox names known
    assert lint["G1_02"]["ruff_F"] == 2 and lint["G1_02"]["pylint_score"] < 5
    assert lint["G1_03"] is None                                                   # no code ran
    cached = S.lint_run(results, tmp_path)                                          # cache hit, same values
    assert cached["G1_02"]["ruff_F"] == 2
    s = S.summarize(results, lint)
    assert s["E3_lint"]["tasks_linted"] == 2 and s["E3_lint"]["ruff_F_total"] == 2
