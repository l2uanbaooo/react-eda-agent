"""Hand-written ReAct agent (no AgentExecutor, no output parser library).

One model reply = ONE action: a ```python block (executed through the LangChain
tool `python_repl`, which wraps the sandbox) or a FINAL: line.
Self-debug: after a failed execution the model gets a short fix prompt (error,
traceback tail, failed code, hint by error type); at most cfg.max_retries
consecutive fixes. Reflexion (cfg.reflexion): lessons from earlier errors of the
same task are added to every later fix prompt.
Every step is written as one JSONL line (thought, code, output, error, retry
index, timings, sandbox, config hash).

CLI (needs the LLM: set EDA_OLLAMA_URL; Docker running for mode A):
    .venv\\Scripts\\python.exe agent.py --task G2_01
    .venv\\Scripts\\python.exe agent.py --file synthetic/sales.csv --question "..." --answer-format "a number"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import prompts as P
from config import PROJECT_ROOT, load_config
from data import profile_file
from llm import LLMError, OllamaClient
from sandbox import create_sandbox
from sandbox.base import ExecResult, format_observation, short_traceback
from sandbox.tool import make_python_repl_tool

_CODE_RE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)
# "FINAL:" at the start of a line, tolerating markdown decoration: **FINAL:** / > FINAL: / ## FINAL:
_FINAL_RE = re.compile(r"^[ \t>*_#]*FINAL\s*:[*_ \t]*(.*)\Z", re.S | re.M | re.I)


@dataclass
class Action:
    kind: str  # "code" | "final" | "invalid"
    thought: str = ""
    code: str = ""
    final_text: str = ""


def parse_action(text: str) -> Action:
    """First complete ```python block wins (the model must SEE results before
    concluding); otherwise a FINAL: line; otherwise the reply is invalid."""
    m = _CODE_RE.search(text or "")
    if m and m.group(1).strip():
        return Action("code", thought=text[:m.start()].strip(), code=m.group(1).strip("\n"))
    f = _FINAL_RE.search(text or "")
    if f:
        return Action("final", thought=text[:f.start()].strip(), final_text=f.group(1).strip())
    return Action("invalid", thought=(text or "").strip())


class JsonlLogger:
    def __init__(self, path: str | Path | None, **base):
        self.base = base
        self.path = Path(path) if path else None
        self._f = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "a", encoding="utf-8", newline="\n")

    def log(self, event: str, **fields) -> None:
        if self._f is None:
            return
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               **self.base, "event": event, **fields}
        self._f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None


@dataclass
class AgentResult:
    status: str  # final | retry_exhausted | max_steps | format_error | llm_error
    final_text: str = ""
    answer_found: bool = False
    answer: object = None
    answer_error: str | None = None
    steps: int = 0
    n_exec: int = 0
    n_exec_errors: int = 0
    n_format_errors: int = 0
    n_repeated_code: int = 0     # identical code re-sent after a successful run (not re-executed)
    incidents: list = field(default_factory=list)
    max_consecutive_errors: int = 0
    successful_code: list = field(default_factory=list)
    lessons: list = field(default_factory=list)
    duration_s: float = 0.0
    llm_tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    sandbox: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Step:
    reply: str
    action: Action
    feedback: str = ""          # user message sent back after this step
    summary: str = ""           # one line used when the step is trimmed from history


class ReActAgent:
    def __init__(self, cfg, llm, sandbox, logger: JsonlLogger | None = None):
        self.cfg, self.llm, self.sandbox = cfg, llm, sandbox
        self.log = logger or JsonlLogger(None)
        self._last: ExecResult | None = None
        self.tool = make_python_repl_tool(sandbox, max_chars=cfg.obs_max_chars, on_result=self._capture)

    def _capture(self, res: ExecResult) -> None:
        self._last = res

    def _messages(self, system: str, task: str, steps: list[_Step]) -> list[dict]:
        keep = max(1, self.cfg.history_keep_steps)
        while True:
            old, recent = steps[:-keep] if len(steps) > keep else [], steps[-keep:]
            first = task if not old else task + "\n\n" + P.history_summary([s.summary for s in old])
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": first}]
            for s in recent:
                msgs += [{"role": "assistant", "content": s.reply}, {"role": "user", "content": s.feedback}]
            size = sum(len(m["content"]) for m in msgs)
            if size <= self.cfg.prompt_char_budget() or keep == 1:
                return msgs
            keep -= 1

    def run(self, question: str, data_path: str, answer_format: str, extra: str = "",
            profile_text: str | None = None) -> AgentResult:
        cfg, t0 = self.cfg, time.perf_counter()
        res = AgentResult(status="max_steps", sandbox=self.sandbox.name)
        system = P.system_prompt(cfg.sandbox_default_sheet)
        task = P.task_prompt(question, data_path, answer_format, extra,
                             profile_text if cfg.profiling else None)
        self.sandbox.run("answer = None")  # a reused session must not leak an old answer
        self.log.log("start", question=question, data_path=data_path, answer_format=answer_format,
                     profiling=cfg.profiling, reflexion=cfg.reflexion, system_chars=len(system),
                     task_chars=len(task), config=cfg.to_dict(), sandbox_info=self.sandbox.info(),
                     versions=self.sandbox.versions())
        steps: list[_Step] = []
        consecutive = format_errors = 0
        exec_ok: list[bool] = []
        last_ok_code, status = None, P.answer_status(False, None)
        for i in range(1, cfg.max_steps + 1):
            res.steps = i
            msgs = self._messages(system, task, steps)
            try:
                out = self.llm.chat(msgs)
            except LLMError as e:
                res.status, res.error = "llm_error", str(e)
                self.log.log("llm_error", step=i, error=str(e))
                break
            res.llm_tokens["prompt"] += out.prompt_tokens or 0
            res.llm_tokens["completion"] += out.completion_tokens or 0
            act = parse_action(out.text)
            rec = {"step": i, "kind": act.kind, "thought": act.thought[:2000], "code": act.code,
                   "llm": {"duration_s": out.duration_s, "attempts": out.attempts,
                           "prompt_tokens": out.prompt_tokens, "completion_tokens": out.completion_tokens,
                           "prompt_chars": sum(len(m["content"]) for m in msgs)}}
            if act.kind == "final":
                found, value, err = self.sandbox.get_answer()
                res.status, res.final_text = "final", act.final_text
                res.answer_found, res.answer, res.answer_error = found and err is None, value, err
                self.log.log("step", **rec, final_text=act.final_text, answer_found=res.answer_found,
                             answer=value, answer_error=err)
                break
            if act.kind == "invalid":
                format_errors += 1
                res.n_format_errors += 1
                self.log.log("step", **rec, format_error=True, raw_reply=out.text[:2000])
                if format_errors > cfg.max_format_errors:
                    res.status = "format_error"
                    break
                steps.append(_Step(out.text, act, P.FORMAT_REMINDER, f"step {i}: invalid reply (no code/FINAL)"))
                continue
            format_errors = 0
            norm = " ".join(act.code.split())
            if norm == last_ok_code:  # repetition loop guard (seen with Qwen on G4_01)
                res.n_repeated_code += 1
                self.log.log("step", **rec, repeated_code=True)
                steps.append(_Step(out.text, act, P.repeat_prompt(status), f"step {i}: repeated the same code"))
                continue
            self._last = None
            observation = self.tool.invoke({"code": act.code})
            r = self._last
            res.n_exec += 1
            exec_ok.append(r.ok)
            flat = " ".join(act.code.split())[:100]
            if r.ok:
                consecutive, last_ok_code = 0, norm
                res.successful_code.append(act.code)
                found, value, err = self.sandbox.get_answer()
                status = P.answer_status(found and err is None, value)
                feedback = P.observation_prompt(observation, status)
                summary = f"step {i}: ran `{flat}` -> ok: {' '.join(r.stdout.split())[:150]}"
            else:
                consecutive, last_ok_code = consecutive + 1, None
                res.n_exec_errors += 1
                res.max_consecutive_errors = max(res.max_consecutive_errors, consecutive)
                kind = "timeout" if r.timed_out else "exec_error"
                res.incidents.append({"exec": res.n_exec, "step": i, "type": kind, "error": r.error,
                                      "state_lost": r.state_lost})
                hint = P.error_hint(r.error, cfg.step_timeout_s)
                earlier = list(res.lessons) if cfg.reflexion else []
                if cfg.reflexion:
                    res.lessons.append(P.lesson(i, r.error, hint))
                feedback = P.fix_prompt(act.code, r.error, short_traceback(r.traceback, 8), consecutive,
                                        cfg.max_retries, hint, earlier, r.state_lost)
                summary = f"step {i}: ran `{flat}` -> FAILED: {(r.error or '')[:150]}"
            self.log.log("step", **rec, exec={k: v for k, v in r.to_dict().items() if k != "stdout"},
                         stdout=r.stdout[:4000], observation=observation, consecutive_errors=consecutive,
                         retry_index=consecutive if not r.ok else 0)
            if consecutive > cfg.max_retries:
                res.status = "retry_exhausted"
                break
            steps.append(_Step(out.text, act, feedback, summary))
        for inc in res.incidents:  # recovered = an OK execution within the next max_retries executions
            nxt = exec_ok[inc["exec"]: inc["exec"] + cfg.max_retries]
            inc["recovered"] = any(nxt)
        res.duration_s = round(time.perf_counter() - t0, 3)
        self.log.log("end", **{k: v for k, v in res.to_dict().items() if k != "successful_code"})
        return res


# ----------------------------------------------------------------------------
# Running one task (used by the CLI and by eval/run_eval.py in GD5)
# ----------------------------------------------------------------------------
@dataclass
class TaskSpec:
    task_id: str
    question: str
    data_rel: str          # relative to datasets/
    answer_format: str
    extra: str = ""


def spec_from_task(task: dict) -> TaskSpec:
    extra = []
    if task.get("sheet"):
        extra.append(f"Use the sheet named {task['sheet']!r}.")
    if task["checker"] == "plot":
        fig = task["params"]["figure"]
        extra.append(f"Build the chart with plotly, save it with save_fig(fig, {fig!r}) "
                     f"and set answer = {fig!r}.")
    return TaskSpec(task["id"], task["question"], task["file"], task["answer_format"], " ".join(extra))


def run_spec(cfg, llm, spec: TaskSpec, run_dir: str | Path) -> tuple[AgentResult, Path, Path]:
    """Fresh sandbox per task. Returns (result, log path, output dir with figures)."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "outputs" / spec.task_id
    log_path = run_dir / "logs" / f"{spec.task_id}.jsonl"
    data_dir = Path(cfg.datasets_dir)
    profile = profile_file(data_dir / spec.data_rel, max_chars=cfg.profile_max_chars) if cfg.profiling else None
    logger = JsonlLogger(log_path, run_id=run_dir.name, task_id=spec.task_id,
                         config_hash=cfg.config_hash(), experiment_hash=cfg.experiment_hash(),
                         sandbox=cfg.sandbox_mode, model=cfg.model)
    try:
        with create_sandbox(cfg, data_dir, out_dir) as sb:
            agent = ReActAgent(cfg, llm, sb, logger)
            result = agent.run(spec.question, sb.data_path(spec.data_rel), spec.answer_format,
                               spec.extra, profile)
    finally:
        logger.close()
    return result, log_path, out_dir


def _make_llm(cfg):
    return OllamaClient(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the ReAct EDA agent on one task")
    ap.add_argument("--task", help="task id from eval/tasks.json, e.g. G2_01")
    ap.add_argument("--file", help="data file relative to datasets/ (ad-hoc question)")
    ap.add_argument("--question")
    ap.add_argument("--answer-format", default="a JSON-compatible value")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-profiling", action="store_true")
    ap.add_argument("--no-reflexion", action="store_true")
    args = ap.parse_args(argv)

    over = {}
    if args.no_profiling:
        over["profiling"] = False
    if args.no_reflexion:
        over["reflexion"] = False
    cfg = load_config(**over)
    sys.path.insert(0, str(PROJECT_ROOT / "eval"))
    import checkers as C

    task = None
    if args.task:
        tasks = json.loads((PROJECT_ROOT / "eval" / "tasks.json").read_text(encoding="utf-8"))["tasks"]
        task = next((t for t in tasks if t["id"] == args.task), None)
        if task is None:
            print(f"unknown task {args.task}", file=sys.stderr)
            return 2
        spec = spec_from_task(task)
    elif args.file and args.question:
        spec = TaskSpec("adhoc", args.question, args.file, args.answer_format)
    else:
        ap.error("give --task, or --file and --question")

    llm = _make_llm(cfg)
    health = llm.health(warmup=False) if hasattr(llm, "health") else {"ok": True}
    if not health.get("ok"):
        print(f"LLM not ready: {health.get('error')}  (run kaggle/health_check.py)", file=sys.stderr)
        return 3
    run_id = args.run_id or f"adhoc-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    result, log_path, out_dir = run_spec(cfg, llm, spec, Path(cfg.runs_dir) / run_id)
    print(f"status={result.status} steps={result.steps} exec={result.n_exec} "
          f"errors={result.n_exec_errors} sandbox={result.sandbox} {result.duration_s:.1f}s")
    print(f"answer_found={result.answer_found} answer={json.dumps(result.answer, ensure_ascii=False)[:500]}")
    print(f"FINAL: {result.final_text[:500]}")
    if task is not None:
        answer = result.answer if result.answer_found else C.NO_ANSWER
        how = "answer"
        if not result.answer_found:
            found, value, how = C.extract_answer_from_final(result.final_text, task["checker"])
            answer = value if found else C.NO_ANSWER
        gt = json.loads((Path(cfg.datasets_dir) / "ground_truth.json").read_text(encoding="utf-8"))
        chk = C.check(task, answer, gt, final_text=result.final_text, artifacts_dir=out_dir)
        print(f"CHECK {task['id']}: {chk.status} ({chk.reason}) [answer source: {how}]")
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
