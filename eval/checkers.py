"""Deterministic answer checkers + answer-text parsing (GD2).

The same functions score the agent (answer read from the sandbox variable
`answer`, or the FINAL: fallback) and ChatGPT (the ANSWER_JSON: line copied
into eval/chatgpt_results.xlsx). No LLM is involved in checking.

Public API
    NO_ANSWER                                  sentinel: no answer could be obtained
    parse_answer_text(text) -> (found, value)  read an "ANSWER_JSON: <json>" line
    extract_answer_from_final(final_text, checker) -> (found, value, how)
    no_causal_claim(text, x_terms, y_terms) -> (ok, offending_sentences)
    resolve(obj, gt)                           replace {"gt": "ds.part.key"} refs
    expected_for(task, gt)                     resolved expected value of a task
    check(task, answer, gt, final_text=None, artifacts_dir=None) -> CheckResult
    CHECKERS                                   name -> checker function
"""
from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


class _NoAnswer:
    def __repr__(self) -> str:
        return "NO_ANSWER"


NO_ANSWER = _NoAnswer()


@dataclass
class CheckResult:
    passed: bool
    status: str  # PASS | FAIL | NO_ANSWER
    reason: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# Answer text parsing (ChatGPT line and FINAL: fallback)
# ----------------------------------------------------------------------------
_ANSWER_LINE_RE = re.compile(r"^\W*ANSWER_JSON\s*:\s*(.*)$", re.IGNORECASE)
# A number token; a sentence-ending "." right after it is allowed ("is 1234.56.").
_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|\.\d)")
_NUMERIC_CHECKERS = {"number", "integer"}


def _strip_code(s: str) -> str:
    s = s.strip().strip("*").strip()  # ChatGPT often bolds the label: **ANSWER_JSON:** 5
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s.strip("`").strip()


def parse_answer_text(text: str | None) -> tuple[bool, Any]:
    """Find the LAST line starting with ANSWER_JSON: and parse its JSON value."""
    if not text:
        return False, None
    for line in reversed(str(text).splitlines()):
        m = _ANSWER_LINE_RE.match(line.strip())
        if not m:
            continue
        payload = _strip_code(m.group(1))
        for candidate in (payload, payload.rstrip(".")):
            try:
                return True, json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
        return False, None
    return False, None


def extract_answer_from_final(final_text: str | None, checker: str) -> tuple[bool, Any, str]:
    """Fallback when the sandbox variable `answer` is missing (spec 3.1).

    1. an ANSWER_JSON: line inside the FINAL text;
    2. numeric checkers only: the text contains exactly one number.
    Nothing else is guessed.
    """
    found, value = parse_answer_text(final_text)
    if found:
        return True, value, "answer_json_line"
    if checker in _NUMERIC_CHECKERS and final_text:
        tokens = _NUMBER_TOKEN_RE.findall(final_text)
        if len(tokens) == 1:
            return True, float(tokens[0].replace(",", "")), "single_number"
    return False, None, "none"


# ----------------------------------------------------------------------------
# Value normalisation
# ----------------------------------------------------------------------------
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


def to_number(x: Any) -> float | None:
    if x is None or isinstance(x, (bool, np.bool_)):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, str):
        s = x.strip().rstrip("%").strip()
        if _THOUSANDS_RE.match(s):
            s = s.replace(",", "")
        try:
            f = float(s)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def norm_text(x: Any, case_sensitive: bool = False) -> str:
    s = " ".join(str(x).split())
    return s if case_sensitive else s.casefold()


def _close(a: float, e: float, tol_abs: float, tol_rel: float) -> bool:
    return abs(a - e) <= max(tol_abs, tol_rel * abs(e))


def _tols(p: dict) -> tuple[float, float]:
    return float(p.get("tol_abs", 1e-9)), float(p.get("tol_rel", 0.0))


# ----------------------------------------------------------------------------
# Ground-truth references
# ----------------------------------------------------------------------------
def resolve_gt(gt: dict, path: str) -> Any:
    """'sales.facts.n_rows_raw' -> gt['datasets']['sales']['facts']['n_rows_raw']."""
    parts = path.split(".")
    node: Any = gt["datasets"]
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"ground-truth path not found: {path} (at '{part}')")
        node = node[part]
    return node


def resolve(obj: Any, gt: dict) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {"gt"}:
            return resolve_gt(gt, obj["gt"])
        return {k: resolve(v, gt) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve(v, gt) for v in obj]
    return obj


def expected_for(task: dict, gt: dict) -> Any:
    exp = task["expected"]
    if isinstance(exp, dict) and set(exp) == {"value"}:
        return resolve(exp["value"], gt)
    return resolve(exp, gt)


# ----------------------------------------------------------------------------
# Scalar / collection checkers: fn(answer, expected, params, ctx) -> (ok, reason, details)
# ----------------------------------------------------------------------------
Outcome = tuple[bool, str, dict]


def chk_number(ans, exp, p, ctx) -> Outcome:
    a = to_number(ans)
    if a is None:
        return False, f"not a number: {ans!r}", {}
    ta, tr = _tols(p)
    ok = _close(a, float(exp), ta, tr)
    return ok, f"answer={a} expected={exp} tol_abs={ta} tol_rel={tr}", {}


def chk_integer(ans, exp, p, ctx) -> Outcome:
    a = to_number(ans)
    if a is None or a != round(a):
        return False, f"not an integer: {ans!r}", {}
    ok = abs(a - float(exp)) <= float(p.get("tol_abs", 0))
    return ok, f"answer={int(a)} expected={exp}", {}


def chk_string(ans, exp, p, ctx) -> Outcome:
    if isinstance(ans, (bool, dict, list)) or ans is None:
        return False, f"not a string: {ans!r}", {}
    cs = bool(p.get("case_sensitive", False))
    accepted = {norm_text(x, cs) for x in [exp, *p.get("aliases", [])]}
    ok = norm_text(ans, cs) in accepted
    return ok, f"answer={ans!r} expected={exp!r}", {}


def chk_set(ans, exp, p, ctx) -> Outcome:
    if not isinstance(ans, (list, tuple)):
        return False, f"not a list: {ans!r}", {}
    cs = bool(p.get("case_sensitive", False))
    a, e = {norm_text(x, cs) for x in ans}, {norm_text(x, cs) for x in exp}
    details = {"missing": sorted(e - a), "extra": sorted(a - e)}
    return a == e and len(ans) == len(a), f"missing={details['missing']} extra={details['extra']}", details


def chk_list(ans, exp, p, ctx) -> Outcome:
    if not isinstance(ans, (list, tuple)):
        return False, f"not a list: {ans!r}", {}
    cs = bool(p.get("case_sensitive", False))
    ok = [norm_text(x, cs) for x in ans] == [norm_text(x, cs) for x in exp]
    return ok, f"answer={list(ans)} expected={exp}", {}


_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}


def _to_bool(x) -> bool | None:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().casefold()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    return None


def chk_bool(ans, exp, p, ctx) -> Outcome:
    a = _to_bool(ans)
    if a is None:
        return False, f"not a boolean: {ans!r}", {}
    return a == bool(exp), f"answer={a} expected={exp}", {}


def chk_datetime(ans, exp, p, ctx) -> Outcome:
    try:
        a, e = pd.Timestamp(str(ans)), pd.Timestamp(str(exp))
    except (ValueError, TypeError):
        return False, f"not a datetime: {ans!r}", {}
    res = p.get("resolution", "s")
    ok = a.floor(res) == e.floor(res)
    return ok, f"answer={a} expected={e}", {}


def chk_dict_number(ans, exp, p, ctx) -> Outcome:
    if not isinstance(ans, dict):
        return False, f"not an object: {ans!r}", {}
    a = {norm_text(k): v for k, v in ans.items()}
    e = {norm_text(k): v for k, v in exp.items()}
    missing, extra = sorted(set(e) - set(a)), sorted(set(a) - set(e))
    if missing or (extra and not p.get("allow_extra_keys", False)):
        return False, f"keys differ: missing={missing} extra={extra}", {"missing": missing, "extra": extra}
    ta, tr = _tols(p)
    bad = {}
    for k, ev in e.items():
        av = to_number(a[k])
        if av is None or not _close(av, float(ev), ta, tr):
            bad[k] = {"answer": a[k], "expected": ev}
    return not bad, f"{len(e) - len(bad)}/{len(e)} values within tolerance", {"wrong": bad}


def chk_composite(ans, exp, p, ctx) -> Outcome:
    """Answer is an object; every field has its own checker (params.fields)."""
    if not isinstance(ans, dict):
        return False, f"not an object: {ans!r}", {}
    lookup = {norm_text(k): v for k, v in ans.items()}
    results, all_ok = {}, True
    for name, spec in p["fields"].items():
        key = norm_text(name)
        if key not in lookup:
            results[name] = {"passed": False, "reason": "missing field"}
            all_ok = False
            continue
        fn = CHECKERS[spec["checker"]]
        ok, reason, det = fn(lookup[key], exp[name], spec.get("params", {}), ctx)
        results[name] = {"passed": ok, "reason": reason, **({"details": det} if det else {})}
        all_ok = all_ok and ok
    n_ok = sum(r["passed"] for r in results.values())
    return all_ok, f"{n_ok}/{len(results)} fields correct", {"fields": results}


# ----------------------------------------------------------------------------
# Correlation is not causation
# ----------------------------------------------------------------------------
_CAUSAL_VERBS = (r"(?:causes?|caused|causing|leads?\s+to|led\s+to|drives?|driving|drove|"
                 r"results?\s+in|increases?|increased|boosts?|makes?|made|triggers?|"
                 r"is\s+responsible\s+for|gây\s+ra|dẫn\s+đến|làm\s+tăng)")
_NEGATION_RE = re.compile(r"\b(?:not|no|never|cannot|neither|nor|without|unlikely)\b|n't|"
                          r"không|chẳng", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def no_causal_claim(text: str, x_terms: list[str], y_terms: list[str]) -> tuple[bool, list[str]]:
    """True if no sentence claims that X causes Y or Y causes X.

    A claim = X-term ... causal verb ... Y-term (either direction) in one
    sentence, with no negation anywhere before the end of that match.
    "Temperature drives both ice cream and fan sales" is NOT a claim.
    """
    bad: list[str] = []
    for sent in _SENTENCE_SPLIT_RE.split(text or ""):
        for a_terms, b_terms in ((x_terms, y_terms), (y_terms, x_terms)):
            a = "|".join(re.escape(t) for t in a_terms)
            b = "|".join(re.escape(t) for t in b_terms)
            pat = re.compile(rf"(?:{a}).{{0,80}}?\b{_CAUSAL_VERBS}\b.{{0,80}}?(?:{b})",
                             re.IGNORECASE | re.DOTALL)
            hit = False
            for m in pat.finditer(sent):
                if not _NEGATION_RE.search(sent[:m.end()]):
                    hit = True
                    break
            if hit:
                bad.append(sent.strip())
                break
    return not bad, bad


def chk_causal(ans, exp, p, ctx) -> Outcome:
    ok, reason, det = chk_composite(ans, exp, p, ctx)
    text = ctx.get("final_text")
    if not text:
        det["text_check"] = "skipped (no final text)"
        return ok, reason, det
    text_ok, offending = no_causal_claim(text, p["x_terms"], p["y_terms"])
    det["text_check"] = "pass" if text_ok else "causal claim found"
    det["offending_sentences"] = offending
    if not text_ok:
        reason = f"{reason}; final text claims causation"
    return ok and text_ok, reason, det


# ----------------------------------------------------------------------------
# Plot checker (reads Plotly figure JSON written by save_fig)
# ----------------------------------------------------------------------------
def decode_array(v: Any) -> list | None:
    """Plotly JSON arrays are lists, or {"dtype", "bdata", "shape"} in plotly>=6."""
    if v is None:
        return None
    if isinstance(v, dict) and "bdata" in v:
        arr = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v.get("dtype", "f8")))
        if v.get("shape"):
            arr = arr.reshape(tuple(int(s) for s in str(v["shape"]).split(",")))
        return arr.tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def load_figure(path: Path) -> tuple[list[dict], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("data", [])), dict(data.get("layout", {}))


def trace_kinds(t: dict) -> set[str]:
    typ = t.get("type", "scatter")
    if typ in ("scatter", "scattergl"):
        mode = t.get("mode")
        if mode is None:  # plotly.js default
            n = len(decode_array(t.get("x")) or decode_array(t.get("y")) or [])
            mode = "lines+markers" if n < 20 else "lines"
        kinds = set()
        if "lines" in mode:
            kinds.add("line")
        if "markers" in mode:
            kinds.add("scatter")
        return kinds
    return {typ}


def _cat_val(t: dict) -> tuple[list, list]:
    x, y = decode_array(t.get("x")) or [], decode_array(t.get("y")) or []
    if t.get("orientation") == "h":
        return y, x
    return x, y


_YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def norm_x(values: list, kind: str) -> list | None:
    if kind in ("date_month", "date_day"):
        # "YYYY-MM" -> "YYYY-MM-01": dateutil would fill the day from today's date.
        strs = [f"{v}-01" if _YEAR_MONTH_RE.match(str(v)) else str(v) for v in values]
        ts = pd.to_datetime(pd.Series(strs), format="mixed", errors="coerce")
        if ts.isna().any():
            return None
        return [str(p) for p in ts.dt.to_period("M" if kind == "date_month" else "D")]
    if kind == "number":
        out = [to_number(v) for v in values]
        return None if any(v is None for v in out) else out
    return [norm_text(v) for v in values]


def _compare_series(t: dict, exp: dict, p: dict, label: str) -> tuple[bool, str]:
    kind = p.get("x_kind", "category")
    cats, vals = _cat_val(t)
    if len(cats) != len(vals) or not cats:
        return False, f"{label}: empty or mismatched x/y"
    keys = norm_x(cats, kind)
    if keys is None:
        return False, f"{label}: x values not parseable as {kind}"
    if len(set(keys)) != len(keys):
        return False, f"{label}: duplicate x values (not aggregated)"
    if p.get("x_sorted") and keys != sorted(keys):
        return False, f"{label}: x axis not sorted ascending"
    exp_keys = norm_x(list(exp.keys()), kind)
    expected = dict(zip(exp_keys, exp.values()))
    if set(keys) != set(expected):
        miss = sorted(set(expected) - set(keys))[:5]
        extra = sorted(set(keys) - set(expected))[:5]
        return False, f"{label}: x categories differ (missing {miss}, extra {extra})"
    tol = float(p.get("y_tol_rel", 0.01))
    nums = [to_number(v) for v in vals]
    if any(v is None for v in nums):
        return False, f"{label}: non-numeric y values"
    wrong = [k for k, v in zip(keys, nums) if not _close(v, float(expected[k]), 1e-9, tol)]
    if wrong:
        return False, f"{label}: {len(wrong)} y values off by more than {tol:.1%} (e.g. {wrong[:3]})"
    if p.get("y_sorted_desc") and nums != sorted(nums, reverse=True):
        return False, f"{label}: bars not sorted descending"
    return True, f"{label}: {len(keys)} points match"


def chk_plot(ans, exp, p, ctx) -> Outcome:
    d = ctx.get("artifacts_dir")
    if d is None:
        return False, "no artifacts_dir given", {}
    path = Path(d) / f"{p['figure']}.json"
    if not path.exists() and isinstance(ans, str) and (Path(d) / f"{ans}.json").exists():
        path = Path(d) / f"{ans}.json"
    if not path.exists():
        return False, f"figure file {path.name} not found", {}
    traces, _layout = load_figure(path)
    want = p["trace_type"]
    typed = [t for t in traces if want in trace_kinds(t)]
    det = {"figure": path.name, "n_traces": len(traces), "n_typed": len(typed)}
    if not typed:
        kinds = sorted({k for t in traces for k in trace_kinds(t)})
        return False, f"no '{want}' trace (found {kinds})", det
    if "n_traces" in p and len(typed) != int(p["n_traces"]):
        return False, f"expected {p['n_traces']} '{want}' traces, found {len(typed)}", det
    if "trace_names" in p:
        names = {norm_text(t.get("name", "")) for t in typed}
        if names != {norm_text(n) for n in p["trace_names"]}:
            return False, f"trace names {sorted(names)} != {sorted(p['trace_names'])}", det

    if want == "histogram":
        t = typed[0]
        raw = decode_array(t.get("x")) if t.get("x") is not None else decode_array(t.get("y"))
        nums = [to_number(v) for v in raw or []]
        nums = [v for v in nums if v is not None]
        if len(nums) != int(p["expected_n"]):
            return False, f"histogram has {len(nums)} values, expected {p['expected_n']}", det
        mean = float(np.mean(nums))
        ok = _close(mean, float(p["expected_mean"]), float(p.get("mean_tol_abs", 0.05)), 0.0)
        return ok, f"n={len(nums)} mean={mean:.4f} expected_mean={p['expected_mean']}", det

    if "expected_corr" in p:  # scatter relationship
        t = typed[0]
        x = [to_number(v) for v in decode_array(t.get("x")) or []]
        y = [to_number(v) for v in decode_array(t.get("y")) or []]
        if not x or len(x) != len(y) or any(v is None for v in x + y):
            return False, "scatter x/y empty, mismatched or non-numeric", det
        if "expected_n" in p and len(x) != int(p["expected_n"]):
            return False, f"scatter has {len(x)} points, expected {p['expected_n']}", det
        r = float(np.corrcoef(x, y)[0, 1])
        xm = float(np.mean(x))
        ok_r = _close(r, float(p["expected_corr"]), float(p.get("corr_tol_abs", 0.02)), 0.0)
        ok_x = ("expected_x_mean" not in p
                or _close(xm, float(p["expected_x_mean"]), float(p.get("x_mean_tol_abs", 0.5)), 0.0))
        det.update({"corr": r, "x_mean": xm})
        reason = f"n={len(x)} corr={r:.4f} (exp {p['expected_corr']}) x_mean={xm:.3f}"
        if not ok_x:
            reason += f" (exp x_mean {p['expected_x_mean']}: wrong variable on x?)"
        return ok_r and ok_x, reason, det

    if "expected_series_by_trace" in p:
        series = p["expected_series_by_trace"]
        by_name = {norm_text(t.get("name", "")): t for t in typed}
        msgs = []
        for name, s in series.items():
            t = by_name.get(norm_text(name))
            if t is None:
                return False, f"no trace named {name}", det
            ok, msg = _compare_series(t, s, p, f"trace {name}")
            if not ok:
                return False, msg, det
            msgs.append(msg)
        return True, "; ".join(msgs), det

    if len(typed) != 1:
        return False, f"expected a single '{want}' trace, found {len(typed)}", det
    ok, msg = _compare_series(typed[0], p["expected_series"], p, want)
    return ok, msg, det


CHECKERS: dict[str, Callable[..., Outcome]] = {
    "number": chk_number,
    "integer": chk_integer,
    "string": chk_string,
    "set": chk_set,
    "list": chk_list,
    "bool": chk_bool,
    "datetime": chk_datetime,
    "dict_number": chk_dict_number,
    "composite": chk_composite,
    "causal": chk_causal,
    "plot": chk_plot,
}


def check(task: dict, answer: Any, gt: dict, final_text: str | None = None,
          artifacts_dir: str | Path | None = None) -> CheckResult:
    """Score one answer for one task. Never raises on a bad answer."""
    name = task["checker"]
    if answer is NO_ANSWER and name != "plot":
        return CheckResult(False, "NO_ANSWER", "no answer obtained")
    exp = expected_for(task, gt)
    params = resolve(task.get("params", {}), gt)
    ctx = {"final_text": final_text, "artifacts_dir": artifacts_dir}
    try:
        ok, reason, details = CHECKERS[name](answer, exp, params, ctx)
    except (TypeError, ValueError, KeyError, AttributeError, IndexError) as e:
        ok, reason, details = False, f"checker error on this answer: {type(e).__name__}: {e}", {}
    return CheckResult(ok, "PASS" if ok else "FAIL", reason, details)
