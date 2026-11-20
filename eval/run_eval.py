"""Evaluation runner: the agent on eval/tasks.json, one result file per task, resumable.

Usage (project root, PowerShell; EDA_OLLAMA_URL set, Docker running):
    .venv\\Scripts\\python.exe eval\\run_eval.py --run-id main-v1
    .venv\\Scripts\\python.exe eval\\run_eval.py --run-id main-v1 --tasks "G1_*,G5_01"
    .venv\\Scripts\\python.exe eval\\run_eval.py --run-id main-v1 --subset chatgpt
    .venv\\Scripts\\python.exe eval\\run_eval.py --run-id main-v1 --score     # then compute metrics

Resume: re-run the SAME command. Tasks with a result file are skipped; tasks
that ended with an infrastructure error (LLM unreachable) are run again.
The run folder holds:
    manifest.json            experiment hash, sandbox mode, model, git commit, data/task hashes
    results/<task_id>.json   outcome + checker verdict (written atomically after each task)
    logs/<task_id>.jsonl     step-by-step agent log
    outputs/<task_id>/       figures saved by save_fig
A run never mixes experiments: if the experiment hash (config without tunnel
URL/paths/timeouts) differs from the manifest, the runner refuses to continue.

Exit codes: 0 done, 3 LLM not ready at start, 4 stopped after an LLM outage
(resume later), 5 sandbox problem (e.g. Docker not running), 6 config mismatch.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import checkers as C  # noqa: E402
from agent import run_spec, spec_from_task  # noqa: E402
from config import load_config  # noqa: E402
from llm import OllamaClient  # noqa: E402
from sandbox import SandboxError  # noqa: E402

TASKS_FILE = ROOT / "eval" / "tasks.json"


class RunConfigMismatch(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)  # atomic on Windows and Linux: no half-written result after a crash


def load_tasks(path: Path = TASKS_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["tasks"]


def select_tasks(tasks: list[dict], patterns: str | None = None, subset: str | None = None) -> list[dict]:
    out = tasks
    if subset == "chatgpt":
        out = [t for t in out if t.get("chatgpt_subset")]
    elif subset:
        raise ValueError(f"unknown subset {subset!r}")
    if patterns:
        pats = [p.strip() for p in patterns.split(",") if p.strip()]
        out = [t for t in out if any(fnmatch.fnmatchcase(t["id"], p) for p in pats)]
    return out


def git_info() -> dict | None:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, timeout=10).stdout.strip() != ""
        return {"commit": head.stdout.strip(), "dirty": dirty}
    except (OSError, subprocess.TimeoutExpired):
        return None


def prepare_manifest(run_dir: Path, cfg, task_ids: list[str], force: bool = False) -> dict:
    gt_path = Path(cfg.datasets_dir) / "ground_truth.json"
    current = {
        "experiment_hash": cfg.experiment_hash(),
        "experiment": cfg.experiment_dict(),
        "sandbox": cfg.sandbox_mode,
        "model": cfg.model,
        "tasks_sha256": _sha256(TASKS_FILE),
        "ground_truth_sha256": _sha256(gt_path),
    }
    path = run_dir / "manifest.json"
    if path.exists():
        man = json.loads(path.read_text(encoding="utf-8"))
        diffs = [k for k in ("experiment_hash", "tasks_sha256", "ground_truth_sha256") if man.get(k) != current[k]]
        if diffs and not force:
            changed = {k: v for k, v in current["experiment"].items() if man.get("experiment", {}).get(k) != v}
            raise RunConfigMismatch(f"run {run_dir.name} was started with a different setup ({', '.join(diffs)}); "
                                    f"changed settings: {changed}. Use a new --run-id (or --force-config).")
        man["tasks"] = sorted(set(man.get("tasks", [])) | set(task_ids))
        man.setdefault("resumed_at", []).append(_now())
        if diffs:
            man.setdefault("forced_mismatch", []).append({"at": _now(), "fields": diffs, "now": current})
    else:
        man = {"run_id": run_dir.name, "created": _now(), **current, "tasks": sorted(task_ids),
               "git": git_info(), "python": sys.version.split()[0], "platform": platform.platform()}
    atomic_write_json(path, man)
    return man


def resolve_answer(task: dict, res) -> tuple[object, str]:
    """Answer used for checking + where it came from (spec 3.1)."""
    if res.status != "final":
        return C.NO_ANSWER, "none"
    if res.answer_found:
        return res.answer, "answer"
    found, value, how = C.extract_answer_from_final(res.final_text, task["checker"])
    return (value, how) if found else (C.NO_ANSWER, "none")


def error_codes(res, source: str) -> dict:
    """Error taxonomy of spec 3.2 (counts per task)."""
    codes: dict[str, int] = {}
    for inc in res.incidents:
        codes[inc["type"]] = codes.get(inc["type"], 0) + 1
    if res.n_format_errors:
        codes["format_error"] = res.n_format_errors
    if res.n_repeated_code:
        codes["repeated_code"] = res.n_repeated_code
    if res.status in ("retry_exhausted", "max_steps", "llm_error"):
        codes[res.status] = 1
    if res.status == "format_error":
        codes["format_error_stop"] = 1
    if res.status == "final" and not res.answer_found:
        codes["answer_missing" if source != "none" else "answer_unparseable"] = 1
    return codes


def evaluate_task(cfg, llm, task: dict, run_dir: Path, gt: dict) -> dict:
    t0 = time.perf_counter()
    res, log_path, out_dir = run_spec(cfg, llm, spec_from_task(task), run_dir)
    infra = res.status == "llm_error"
    answer, source = resolve_answer(task, res)
    chk = None if infra else C.check(task, answer, gt, final_text=res.final_text, artifacts_dir=out_dir)
    return {
        "task_id": task["id"], "group": task["group"], "checker": task["checker"],
        "chatgpt_subset": bool(task.get("chatgpt_subset")),
        "status": res.status, "infra_error": infra,
        "passed": bool(chk and chk.passed),
        "check": chk.to_dict() if chk else None,
        "answer": res.answer, "answer_found": res.answer_found, "answer_error": res.answer_error,
        "answer_source": source, "final_text": res.final_text,
        "error_codes": error_codes(res, source),
        "incidents": res.incidents, "steps": res.steps, "n_exec": res.n_exec,
        "n_exec_errors": res.n_exec_errors, "n_format_errors": res.n_format_errors,
        "n_repeated_code": res.n_repeated_code, "max_consecutive_errors": res.max_consecutive_errors,
        "lessons": res.lessons, "successful_code": res.successful_code,
        "llm_tokens": res.llm_tokens, "duration_s": round(time.perf_counter() - t0, 3),
        "agent_duration_s": res.duration_s, "agent_error": res.error,
        "sandbox": res.sandbox, "model": cfg.model,
        "experiment_hash": cfg.experiment_hash(), "config_hash": cfg.config_hash(),
        "log": str(log_path.relative_to(run_dir)), "outputs": str(out_dir.relative_to(run_dir)),
        "finished_at": _now(),
    }


def _wait_for_llm(llm, cfg, sleep, log) -> bool:
    for k in range(1, cfg.eval_outage_checks + 1):
        log(f"   LLM unreachable: waiting {cfg.eval_outage_wait_s:g} s (check {k}/{cfg.eval_outage_checks})")
        sleep(cfg.eval_outage_wait_s)
        if llm.health(warmup=False).get("ok"):
            log("   LLM is back")
            return True
    return False


def run_eval(cfg, tasks: list[dict], run_id: str, llm=None, force_config: bool = False,
             sleep=time.sleep, log=print) -> dict:
    run_dir = Path(cfg.runs_dir) / run_id
    results_dir = run_dir / "results"
    try:
        prepare_manifest(run_dir, cfg, [t["id"] for t in tasks], force=force_config)
    except RunConfigMismatch as e:
        log(f"REFUSED: {e}")
        return {"exit_code": 6, "stopped": "config_mismatch"}
    gt = json.loads((Path(cfg.datasets_dir) / "ground_truth.json").read_text(encoding="utf-8"))
    llm = llm or OllamaClient(cfg)
    health = llm.health(warmup=True)
    if not health.get("ok"):
        log(f"LLM not ready: {health.get('error')} -> run kaggle/health_check.py, then re-run this command")
        return {"exit_code": 3, "stopped": "llm_not_ready"}

    def done(t):
        p = results_dir / f"{t['id']}.json"
        return p.exists() and not json.loads(p.read_text(encoding="utf-8")).get("infra_error")

    todo = [t for t in tasks if not done(t)]
    log(f"run {run_id}: {len(tasks)} selected, {len(tasks) - len(todo)} already done, {len(todo)} to run "
        f"(sandbox={cfg.sandbox_mode}, experiment={cfg.experiment_hash()})")
    n_run = 0
    for i, task in enumerate(todo, start=1):
        log(f"[{i}/{len(todo)}] {task['id']} ...")
        for attempt in (1, 2):
            try:
                result = evaluate_task(cfg, llm, task, run_dir, gt)
            except SandboxError as e:
                log(f"SANDBOX PROBLEM: {e}")
                return {"exit_code": 5, "stopped": "sandbox_error", "ran": n_run, "error": str(e)}
            atomic_write_json(results_dir / f"{task['id']}.json", result)
            if not result["infra_error"]:
                break
            if attempt == 2 or not _wait_for_llm(llm, cfg, sleep, log):
                log("LLM still unreachable: stopping. Restart the Kaggle tunnel, set EDA_OLLAMA_URL, "
                    "and re-run the same command to resume.")
                return {"exit_code": 4, "stopped": "llm_outage", "ran": n_run}
        n_run += 1
        verdict = "PASS" if result["passed"] else ("INFRA" if result["infra_error"] else "FAIL")
        log(f"   -> {verdict} status={result['status']} exec={result['n_exec']} "
            f"errors={result['n_exec_errors']} answer_source={result['answer_source']} "
            f"{result['duration_s']:.0f}s")
    return {"exit_code": 0, "stopped": None, "ran": n_run, "run_dir": str(run_dir)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the agent on the task set (resumable)")
    ap.add_argument("--run-id", default=None, help="folder name under runs/ (reuse it to resume)")
    ap.add_argument("--tasks", default=None, help="comma-separated ids or globs, e.g. 'G1_*,G5_01'")
    ap.add_argument("--subset", default=None, choices=["chatgpt"], help="only the 12 ChatGPT-comparison tasks")
    ap.add_argument("--force-config", action="store_true", help="resume despite a different experiment hash")
    ap.add_argument("--score", action="store_true", help="run eval/score.py at the end")
    args = ap.parse_args(argv)
    cfg = load_config()
    tasks = select_tasks(load_tasks(), args.tasks, args.subset)
    if not tasks:
        print("no task selected", file=sys.stderr)
        return 2
    run_id = args.run_id or f"eval-{datetime.now():%Y%m%d-%H%M}"
    out = run_eval(cfg, tasks, run_id, force_config=args.force_config)
    if out["exit_code"] in (0, 4):
        print(f"resume/continue with: .venv\\Scripts\\python.exe eval\\run_eval.py --run-id {run_id}"
              + (f" --tasks \"{args.tasks}\"" if args.tasks else "")
              + (f" --subset {args.subset}" if args.subset else ""))
    if args.score and out["exit_code"] == 0:
        import score
        return score.main(["--run-id", run_id])
    return out["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
