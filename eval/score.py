"""Metrics of one eval run (spec 2.2/2.3). Numbers come ONLY from result files.

Usage:
    .venv\\Scripts\\python.exe eval\\score.py --run-id main-v1
    .venv\\Scripts\\python.exe eval\\score.py --run-id main-v1 --no-lint

Writes runs/<run_id>/summary.json and summary.md.

Definitions (fixed in SPEC_GD0.md before any result existed):
  K1  task completion = PASS / scored tasks; infra errors (LLM unreachable)
      are excluded from the denominator and reported separately.
      K1-strict counts only answers read from the `answer` variable.
  E2  retries = failed executions per task (+ distribution), steps per task.
  K2  self-debug = incidents (failed execution or timeout) followed by an OK
      execution within the next 3 executions / all incidents.
      K2-strict additionally requires the task to PASS.
  E3  lint = pylint score (eval/pylintrc) and ruff F-rule count on the
      successfully executed code of each task.
All rates come with a 95% Wilson interval; small samples are flagged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import load_config  # noqa: E402

PYLINTRC = ROOT / "eval" / "pylintrc"
SANDBOX_NAMES = ["pd", "np", "px", "go", "load_data", "save_fig", "DATA_DIR", "OUTPUT_DIR", "answer"]
TARGETS = {"K1": 0.80, "K2": 0.70}
SMALL_SAMPLE = 20
GROUPS = [f"G{i}" for i in range(1, 7)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(centre - half, 4), round(centre + half, 4)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [lo, hi],
            "small_sample": n < SMALL_SAMPLE}


def load_results(run_dir: Path) -> list[dict]:
    files = sorted((run_dir / "results").glob("*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


# ----------------------------------------------------------------------------
# Lint (E3)
# ----------------------------------------------------------------------------
def task_code(result: dict) -> str:
    return "\n\n".join(f"# --- step {i} ---\n{c}" for i, c in enumerate(result.get("successful_code", []), 1))


def lint_code(code: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    pl = subprocess.run([sys.executable, "-m", "pylint", f"--rcfile={PYLINTRC}", "--output-format=json2",
                         "--score=y", str(path)], capture_output=True, text=True, encoding="utf-8",
                        timeout=300, cwd=path.parent)
    try:
        report = json.loads(pl.stdout)
        score = report["statistics"]["score"]
        messages = report["messages"]
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"pylint failed on {path.name}: {pl.stderr[-500:]}") from e
    # ruff 0.8: `builtins` is a top-level option (not lint.builtins). F811 (re-import) is an
    # artefact of concatenating steps. Exit code 2 = ruff itself failed: never read that as 0 findings.
    builtins = "builtins = [" + ", ".join(f'"{n}"' for n in SANDBOX_NAMES) + "]"
    rf = subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "F", "--ignore", "F811",
                         "--output-format", "json", "--no-cache", "--config", builtins, str(path)],
                        capture_output=True, text=True, encoding="utf-8", timeout=120, cwd=path.parent)
    if rf.returncode not in (0, 1):
        raise RuntimeError(f"ruff failed on {path.name} (exit {rf.returncode}): {rf.stderr[-500:]}")
    ruff_msgs = json.loads(rf.stdout)
    return {"pylint_score": round(max(float(score), 0.0), 2), "pylint_raw_score": float(score),
            "pylint_messages": Counter(m["symbol"] for m in messages).most_common(),
            "ruff_F": len(ruff_msgs), "ruff_codes": Counter(m["code"] for m in ruff_msgs).most_common(),
            "lines": code.count("\n") + 1}


def lint_run(results: list[dict], run_dir: Path) -> dict:
    """Per-task lint, cached by code hash in runs/<id>/lint/lint.json."""
    cache_path = run_dir / "lint" / "lint.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    out = {}
    for r in results:
        code = task_code(r)
        if not code.strip():
            out[r["task_id"]] = None
            continue
        key = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        if cache.get(r["task_id"], {}).get("key") != key:
            cache[r["task_id"]] = {"key": key, **lint_code(code, run_dir / "lint" / f"{r['task_id']}.py")}
        out[r["task_id"]] = cache[r["task_id"]]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def summarize(results: list[dict], lint: dict | None = None) -> dict:
    scored = [r for r in results if not r["infra_error"]]
    infra = [r["task_id"] for r in results if r["infra_error"]]
    passed = [r for r in scored if r["passed"]]
    s: dict = {"n_results": len(results), "n_scored": len(scored), "infra_error_tasks": infra}

    s["K1_task_completion"] = rate(len(passed), len(scored))
    s["K1_strict_answer_variable_only"] = rate(sum(r["answer_source"] == "answer" for r in passed), len(scored))
    s["K1_by_group"] = {g: rate(sum(r["passed"] for r in scored if r["group"] == g),
                                sum(r["group"] == g for r in scored)) for g in GROUPS}
    s["status_counts"] = dict(Counter(r["status"] for r in scored))
    s["answer_source_counts"] = dict(Counter(r["answer_source"] for r in scored))
    tasks_with, occurrences = Counter(), Counter()
    for r in scored:
        for code, n in r["error_codes"].items():
            tasks_with[code] += 1
            occurrences[code] += n
    s["error_codes"] = {c: {"tasks": tasks_with[c], "occurrences": occurrences[c]} for c in sorted(tasks_with)}

    errs = [r["n_exec_errors"] for r in scored]
    dist = Counter("3+" if e >= 3 else str(e) for e in errs)
    s["E2_retries"] = {
        "failed_executions_per_task_mean": round(statistics.mean(errs), 3) if errs else None,
        "failed_executions_per_task_median": statistics.median(errs) if errs else None,
        "distribution": {k: dist.get(k, 0) for k in ("0", "1", "2", "3+")},
        "steps_per_task_mean": round(statistics.mean(r["steps"] for r in scored), 3) if scored else None,
        "executions_per_task_mean": round(statistics.mean(r["n_exec"] for r in scored), 3) if scored else None,
        "repeated_code_total": sum(r.get("n_repeated_code", 0) for r in scored),
    }

    incidents = [(inc, r) for r in scored for inc in r["incidents"]]
    rec = [inc for inc, _ in incidents if inc["recovered"]]
    s["K2_self_debug"] = rate(len(rec), len(incidents))
    s["K2_strict_recovered_and_task_passed"] = rate(sum(inc["recovered"] and r["passed"] for inc, r in incidents),
                                                    len(incidents))
    s["K2_by_type"] = {t: rate(sum(i["recovered"] for i, _ in incidents if i["type"] == t),
                               sum(i["type"] == t for i, _ in incidents)) for t in ("exec_error", "timeout")}

    if lint is not None:
        vals = [v for v in lint.values() if v]
        s["E3_lint"] = {
            "tasks_linted": len(vals),
            "pylint_score_mean": round(statistics.mean(v["pylint_score"] for v in vals), 3) if vals else None,
            "pylint_score_min": min((v["pylint_score"] for v in vals), default=None),
            "ruff_F_total": sum(v["ruff_F"] for v in vals),
            "ruff_F_per_task_mean": round(statistics.mean(v["ruff_F"] for v in vals), 3) if vals else None,
            "top_pylint_messages": Counter({k: n for v in vals for k, n in v["pylint_messages"]}).most_common(8),
        }

    s["duration_s_total"] = round(sum(r["duration_s"] for r in results), 1)
    s["llm_tokens_total"] = {k: sum(r["llm_tokens"].get(k, 0) for r in results) for k in ("prompt", "completion")}
    s["sandboxes"] = sorted({r["sandbox"] for r in results})
    s["models"] = sorted({r["model"] for r in results})
    s["experiment_hashes"] = sorted({r["experiment_hash"] for r in results})
    k1, k2 = s["K1_task_completion"]["rate"], s["K2_self_debug"]["rate"]
    s["targets"] = {
        "K1>=0.80": None if k1 is None else k1 >= TARGETS["K1"],
        "K2>=0.70": None if k2 is None else k2 >= TARGETS["K2"],
    }
    s["tasks"] = [{"task_id": r["task_id"], "group": r["group"], "status": r["status"],
                   "result": "INFRA" if r["infra_error"] else ("PASS" if r["passed"] else "FAIL"),
                   "answer_source": r["answer_source"], "exec_errors": r["n_exec_errors"],
                   "reason": (r["check"] or {}).get("reason", r.get("agent_error") or "")[:160],
                   **({"pylint": (lint or {}).get(r["task_id"], {}) and lint[r["task_id"]]["pylint_score"]}
                      if lint is not None else {})}
                  for r in sorted(results, key=lambda r: r["task_id"])]
    return s


def _fmt_rate(x: dict) -> str:
    if x["rate"] is None:
        return "n/a (no data)"
    lo, hi = x["ci95"]
    warn = " - SMALL SAMPLE" if x["small_sample"] else ""
    return f"{x['rate']:.1%} ({x['k']}/{x['n']}; 95% CI {lo:.1%}-{hi:.1%}){warn}"


def _verdict(v) -> str:
    return "n/a" if v is None else ("MET" if v else "NOT MET")


def to_markdown(run_id: str, s: dict, manifest: dict | None) -> str:
    m = manifest or {}
    lines = [f"# Eval summary: {run_id}", "",
             f"- sandbox: {', '.join(s['sandboxes'])}" + ("  **(MIXED - results not comparable)**"
                                                          if len(s["sandboxes"]) > 1 else ""),
             f"- model: {', '.join(s['models'])}; experiment: {', '.join(s['experiment_hashes'])}",
             f"- git: {(m.get('git') or {}).get('commit', 'n/a')}"
             + (" (dirty)" if (m.get("git") or {}).get("dirty") else ""),
             f"- results: {s['n_results']} tasks, {s['n_scored']} scored, infra errors (excluded): "
             f"{', '.join(s['infra_error_tasks']) or 'none'}", "",
             "| Metric | Value | Target |", "|---|---|---|",
             f"| K1 task completion | {_fmt_rate(s['K1_task_completion'])} | >= 80%: "
             f"{_verdict(s['targets']['K1>=0.80'])} |",
             f"| K1-strict (`answer` variable only) | {_fmt_rate(s['K1_strict_answer_variable_only'])} | |",
             f"| K2 self-debug success | {_fmt_rate(s['K2_self_debug'])} | >= 70%: "
             f"{_verdict(s['targets']['K2>=0.70'])} |",
             f"| K2-strict (recovered and task passed) | {_fmt_rate(s['K2_strict_recovered_and_task_passed'])} | |",
             f"| Failed executions per task (mean) | {s['E2_retries']['failed_executions_per_task_mean']} | |",
             f"| Steps per task (mean) | {s['E2_retries']['steps_per_task_mean']} | |"]
    if "E3_lint" in s:
        lines.append(f"| pylint score (mean, 0-10) | {s['E3_lint']['pylint_score_mean']} | |")
        lines.append(f"| ruff F findings (total) | {s['E3_lint']['ruff_F_total']} | |")
    lines += ["", "## By group", "", "| Group | Completion |", "|---|---|"]
    lines += [f"| {g} | {_fmt_rate(v)} |" for g, v in s["K1_by_group"].items()]
    lines += ["", "## Error codes (tasks / occurrences)", ""]
    lines += [f"- {c}: {v['tasks']} / {v['occurrences']}" for c, v in s["error_codes"].items()] or ["- none"]
    lines += ["", f"Retry distribution (failed executions per task): {s['E2_retries']['distribution']}", "",
              "## Tasks", "", "| Task | Result | Status | Answer source | Exec errors | Reason |",
              "|---|---|---|---|---|---|"]
    for t in s["tasks"]:
        reason = t["reason"].replace("|", "/")
        lines.append(f"| {t['task_id']} | {t['result']} | {t['status']} | {t['answer_source']} | "
                     f"{t['exec_errors']} | {reason} |")
    lines += ["", "Numbers are computed by eval/score.py from the result files only."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compute metrics of an eval run")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--no-lint", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    run_dir = Path(cfg.runs_dir) / args.run_id
    results = load_results(run_dir)
    if not results:
        print(f"no results in {run_dir / 'results'}", file=sys.stderr)
        return 2
    lint = None if args.no_lint else lint_run(results, run_dir)
    s = summarize(results, lint)
    man_path = run_dir / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else None
    s["run_id"], s["manifest"] = args.run_id, manifest
    (run_dir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8", newline="\n")
    md = to_markdown(args.run_id, s, manifest)
    (run_dir / "summary.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
