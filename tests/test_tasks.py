"""GD2 tests for eval/tasks.json: schema, coverage, and for EVERY task a correct
answer must PASS and several wrong answers must FAIL.

Run: .venv\\Scripts\\python.exe -m pytest tests\\test_tasks.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

import checkers as C
import make_chatgpt_template as T

ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "eval" / "tasks.json"
TASKS = json.loads(TASKS_PATH.read_text(encoding="utf-8"))["tasks"]
GT = json.loads((ROOT / "datasets" / "ground_truth.json").read_text(encoding="utf-8"))
BY_ID = {t["id"]: t for t in TASKS}
NON_PLOT = [t for t in TASKS if t["checker"] != "plot"]
GROUPS = [f"G{i}" for i in range(1, 7)]
REQUIRED = {"id", "group", "file", "sheet", "question", "answer_format", "checker",
            "expected", "params", "chatgpt_subset"}


# --------------------------------------------------------------------------
# Schema and coverage
# --------------------------------------------------------------------------
def test_one_task_per_line():
    lines = TASKS_PATH.read_text(encoding="utf-8").splitlines()
    assert sum(1 for ln in lines if ln.startswith('{"id": ')) == len(TASKS)


def test_ids_unique_and_well_formed():
    ids = [t["id"] for t in TASKS]
    assert len(ids) == len(set(ids))
    for t in TASKS:
        assert re.fullmatch(r"G[1-6]_\d{2}", t["id"]), t["id"]
        assert t["id"].startswith(t["group"] + "_")


def test_coverage_six_groups_at_least_30_tasks():
    assert len(TASKS) >= 30
    for g in GROUPS:
        assert sum(t["group"] == g for t in TASKS) >= 5, g


def test_chatgpt_subset_is_two_per_group():
    for g in GROUPS:
        assert sum(t["group"] == g and t["chatgpt_subset"] for t in TASKS) == 2, g


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_task_schema_and_references(task):
    assert REQUIRED <= set(task), REQUIRED - set(task)
    assert task["checker"] in C.CHECKERS
    assert (ROOT / "datasets" / task["file"]).exists(), task["file"]
    C.expected_for(task, GT)          # raises KeyError on a bad gt path
    C.resolve(task["params"], GT)
    if task["checker"] in ("composite", "causal"):
        assert set(task["params"]["fields"]) == set(C.expected_for(task, GT))
        for spec in task["params"]["fields"].values():
            assert spec["checker"] in C.CHECKERS


# --------------------------------------------------------------------------
# Generated correct / wrong answers per checker type
# --------------------------------------------------------------------------
def _round(v, nd):
    return round(float(v), nd) if nd is not None else float(v)


def correct_answer(checker, exp, p):
    if checker == "number":
        return _round(exp, p.get("round"))
    if checker == "integer":
        return int(exp)
    if checker == "string":
        return f"  {str(exp).upper()} "
    if checker == "set":
        return list(reversed(exp))
    if checker == "list":
        return list(exp)
    if checker == "bool":
        return bool(exp)
    if checker == "datetime":
        return pd.Timestamp(exp).isoformat()
    if checker == "dict_number":
        return {str(k).upper(): _round(v, p.get("round")) for k, v in exp.items()}
    if checker in ("composite", "causal"):
        return {name: correct_answer(spec["checker"], exp[name], spec.get("params", {}))
                for name, spec in p["fields"].items()}
    raise AssertionError(checker)


def wrong_answers(checker, exp, p):
    if checker == "number":
        return [float(exp) * 1.2 + 1, "abc", None, True]
    if checker == "integer":
        return [int(exp) + 1, float(exp) + 0.5, "abc"]
    if checker == "string":
        return ["zzz_wrong", None, ["x"]]
    if checker == "set":
        out = [list(exp) + ["bogus"], "not a list"]
        if len(exp) > 1:
            out.append(list(exp)[:-1])
        return out
    if checker == "list":
        return [["bogus"], list(reversed(exp)) if len(exp) > 1 else ["bogus2"]]
    if checker == "bool":
        return [not exp, "maybe"]
    if checker == "datetime":
        return [(pd.Timestamp(exp) + pd.Timedelta(hours=1)).isoformat(), "not a date"]
    if checker == "dict_number":
        good = correct_answer(checker, exp, p)
        k0 = next(iter(good))
        perturbed = {**good, k0: float(good[k0]) * 1.2 + 1}
        missing = {k: v for k, v in good.items() if k != k0}
        extra = {**good, "bogus_key": 1}
        return [perturbed, missing, extra, list(good.values())]
    if checker in ("composite", "causal"):
        good = correct_answer(checker, exp, p)
        out = ["not an object"]
        for name, spec in p["fields"].items():
            bad_value = wrong_answers(spec["checker"], exp[name], spec.get("params", {}))[0]
            out.append({**good, name: bad_value})
            out.append({k: v for k, v in good.items() if k != name})
        return out
    raise AssertionError(checker)


@pytest.mark.parametrize("task", NON_PLOT, ids=lambda t: t["id"])
def test_correct_answer_passes(task):
    exp = C.expected_for(task, GT)
    p = C.resolve(task["params"], GT)
    ans = correct_answer(task["checker"], exp, p)
    res = C.check(task, ans, GT)
    assert res.passed, (res.reason, res.details)


@pytest.mark.parametrize("task", NON_PLOT, ids=lambda t: t["id"])
def test_wrong_answers_fail(task):
    exp = C.expected_for(task, GT)
    p = C.resolve(task["params"], GT)
    for ans in wrong_answers(task["checker"], exp, p):
        res = C.check(task, ans, GT)
        assert not res.passed, f"wrong answer accepted: {ans!r}"


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_no_answer_never_passes(task):
    res = C.check(task, C.NO_ANSWER, GT, artifacts_dir=None)
    assert not res.passed


def test_numeric_tasks_declare_rounding_compatible_with_tolerance():
    # every number/dict_number field must say which rounding the answer_format asks for
    for t in NON_PLOT:
        specs = (t["params"]["fields"].values() if t["checker"] in ("composite", "causal")
                 else [{"checker": t["checker"], "params": t["params"]}])
        for s in specs:
            sp = s.get("params", {})
            if s["checker"] == "number":
                assert "round" in sp, t["id"]
            if s["checker"] == "dict_number" and sp.get("tol_abs", None) != 0:
                assert "round" in sp, t["id"]


# --------------------------------------------------------------------------
# Causal task: structured answer AND final text
# --------------------------------------------------------------------------
def test_causal_task_checks_final_text():
    task = BY_ID["G3_02"]
    exp = C.expected_for(task, GT)
    ans = correct_answer("causal", exp, C.resolve(task["params"], GT))
    good = "Daily ice cream and fan units are correlated, but not causally: both follow temp_c."
    bad = "Selling more ice cream leads to more fan sales."
    assert C.check(task, ans, GT, final_text=good).passed
    assert not C.check(task, ans, GT, final_text=bad).passed
    assert C.check(task, ans, GT, final_text=None).passed  # text check skipped, noted
    assert not C.check(task, {**ans, "is_causal": True}, GT, final_text=good).passed


# --------------------------------------------------------------------------
# Real-data literals must match the downloaded files
# --------------------------------------------------------------------------
def test_titanic_literal_expectations():
    t = pd.read_csv(ROOT / "datasets" / "real" / "titanic.csv")
    exp = C.expected_for(BY_ID["G6_05"], GT)
    by_sex = t.groupby("sex")["survived"].mean() * 100
    by_cls = t.groupby("pclass")["survived"].mean() * 100
    for k, v in exp["by_sex"].items():
        assert v == pytest.approx(by_sex[k], abs=1e-4)
    for k, v in exp["by_pclass"].items():
        assert v == pytest.approx(by_cls[int(k)], abs=1e-4)


def test_housing_literal_expectations():
    h = pd.read_csv(ROOT / "datasets" / "real" / "california_housing.csv")
    exp = C.expected_for(BY_ID["G6_06"], GT)
    assert exp["n_missing_total_bedrooms"] == int(h["total_bedrooms"].isna().sum())
    med = h.groupby("ocean_proximity")["median_house_value"].median()
    assert exp["median_value_by_ocean_proximity"] == pytest.approx(med.to_dict())


# --------------------------------------------------------------------------
# Plot tasks: real Plotly figures, correct and wrong
# --------------------------------------------------------------------------
def _save(fig, d: Path, name: str):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(fig.to_json(), encoding="utf-8")


def _plot_check(task_id, fig, tmp_path, sub):
    task = BY_ID[task_id]
    d = tmp_path / sub
    if fig is not None:
        _save(fig, d, task["params"]["figure"])
    return C.check(task, task["params"]["figure"], GT, artifacts_dir=d)


def _series(task_id):
    return C.resolve(BY_ID[task_id]["params"], GT)


def test_plot_g5_01_monthly_line(tmp_path):
    s = _series("G5_01")["expected_series"]
    keys, vals = list(s.keys()), list(s.values())
    df = pd.DataFrame({"month": pd.PeriodIndex(keys, freq="M").to_timestamp(), "revenue": vals})
    ok = {
        "go_line": go.Figure(go.Scatter(x=keys, y=vals, mode="lines")),
        "px_line_timestamps": px.line(df, x="month", y="revenue"),
        "lines_markers": go.Figure(go.Scatter(x=keys, y=vals, mode="lines+markers")),
    }
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(keys))
    bad = {
        "unsorted": go.Figure(go.Scatter(x=[keys[i] for i in perm], y=[vals[i] for i in perm], mode="lines")),
        "scaled": go.Figure(go.Scatter(x=keys, y=[v * 1.2 for v in vals], mode="lines")),
        "bar_instead": go.Figure(go.Bar(x=keys, y=vals)),
        "one_year_only": go.Figure(go.Scatter(x=keys[12:], y=vals[12:], mode="lines")),
        "not_aggregated": go.Figure(go.Scatter(x=keys + keys[:1], y=vals + vals[:1], mode="lines")),
        "missing_file": None,
    }
    for name, fig in ok.items():
        res = _plot_check("G5_01", fig, tmp_path, name)
        assert res.passed, (name, res.reason)
    for name, fig in bad.items():
        res = _plot_check("G5_01", fig, tmp_path, name)
        assert not res.passed, name


def test_plot_g5_02_bar_by_department(tmp_path):
    s = _series("G5_02")["expected_series"]
    keys, vals = list(s.keys()), list(s.values())
    ok = {
        "vertical": go.Figure(go.Bar(x=keys, y=vals)),
        "horizontal": go.Figure(go.Bar(x=vals, y=keys, orientation="h")),
        "px_bar": px.bar(pd.DataFrame({"dept_name": keys, "salary": vals}), x="dept_name", y="salary"),
    }
    bad = {
        "missing_dept": go.Figure(go.Bar(x=keys[:-1], y=vals[:-1])),
        "orphans_as_unknown": go.Figure(go.Bar(x=keys + ["Unknown"], y=vals + [2000.0])),
        "scaled": go.Figure(go.Bar(x=keys, y=[v * 1.05 for v in vals])),
        "line_instead": go.Figure(go.Scatter(x=keys, y=vals, mode="lines")),
    }
    for name, fig in ok.items():
        assert _plot_check("G5_02", fig, tmp_path, name).passed, name
    for name, fig in bad.items():
        assert not _plot_check("G5_02", fig, tmp_path, name).passed, name


def test_plot_g5_03_one_line_per_sensor(tmp_path):
    by = _series("G5_03")["expected_series_by_trace"]
    rows = [(sid, day, v) for sid, s in by.items() for day, v in s.items()]
    long = pd.DataFrame(rows, columns=["sensor_id", "date", "temp"])
    long["date"] = pd.to_datetime(long["date"])

    def fig_from(series_by, mode="lines", names=None):
        fig = go.Figure()
        for i, (sid, s) in enumerate(series_by.items()):
            fig.add_trace(go.Scatter(x=list(s), y=list(s.values()), mode=mode,
                                     name=(names[i] if names else sid)))
        return fig

    shuffled = dict(by)
    s1 = list(by["S1"].items())
    shuffled["S1"] = dict(s1[1:] + s1[:1])  # first day moved to the end
    ok = {
        "go_traces": fig_from(by),
        "px_color": px.line(long.sort_values("date"), x="date", y="temp", color="sensor_id"),
    }
    bad = {
        "unsorted_s1": fig_from(shuffled),
        "two_sensors": fig_from({k: by[k] for k in ("S1", "S2")}),
        "wrong_names": fig_from(by, names=["sensor 1", "sensor 2", "sensor 3"]),
        "single_line_all": go.Figure(go.Scatter(x=list(by["S1"]), y=list(by["S1"].values()), mode="lines")),
        "markers_only": fig_from(by, mode="markers"),
    }
    for name, fig in ok.items():
        res = _plot_check("G5_03", fig, tmp_path, name)
        assert res.passed, (name, res.reason)
    for name, fig in bad.items():
        assert not _plot_check("G5_03", fig, tmp_path, name).passed, name


def test_plot_g5_04_age_histogram(tmp_path):
    df = pd.read_excel(ROOT / "datasets" / "synthetic" / "clinic_messy.xlsx", sheet_name="patients")
    num = pd.to_numeric(df["age"], errors="coerce")
    valid = num[num.between(0, 120)]
    only_ints = pd.Series([v for v in df["age"] if isinstance(v, (int, float)) and 0 <= v <= 120])
    ok = {
        "go_hist": go.Figure(go.Histogram(x=valid.tolist())),
        "px_hist": px.histogram(pd.DataFrame({"age": valid}), x="age"),
    }
    bad = {
        "invalid_kept": go.Figure(go.Histogram(x=num.dropna().tolist())),
        "text_numbers_dropped": go.Figure(go.Histogram(x=only_ints.tolist())),
        "binned_bar": go.Figure(go.Bar(x=list(range(10)), y=np.histogram(valid, 10)[0].tolist())),
    }
    for name, fig in ok.items():
        res = _plot_check("G5_04", fig, tmp_path, name)
        assert res.passed, (name, res.reason)
    for name, fig in bad.items():
        assert not _plot_check("G5_04", fig, tmp_path, name).passed, name


def test_plot_g5_05_scatter_icecream_vs_temp(tmp_path):
    tx = pd.read_parquet(ROOT / "datasets" / "synthetic" / "transactions.parquet")
    day = tx["ts"].dt.normalize()
    temp = tx.groupby(day)["temp_c"].first()
    is_ice = tx["product_category"] == "ice_cream"
    ice = tx[is_ice].groupby(day[is_ice])["quantity"].sum().reindex(temp.index, fill_value=0)
    daily = pd.DataFrame({"temp_c": temp, "ice_units": ice})
    ok = {
        "go_markers": go.Figure(go.Scatter(x=temp.tolist(), y=ice.tolist(), mode="markers")),
        "px_scatter": px.scatter(daily, x="temp_c", y="ice_units"),
    }
    txi = tx[is_ice]
    bad = {
        "axes_swapped": go.Figure(go.Scatter(x=ice.tolist(), y=temp.tolist(), mode="markers")),
        "transaction_level": go.Figure(go.Scatter(x=txi["temp_c"].tolist(), y=txi["quantity"].tolist(),
                                                  mode="markers")),
        "line_only": go.Figure(go.Scatter(x=temp.tolist(), y=ice.tolist(), mode="lines")),
        "fan_instead": go.Figure(go.Scatter(
            x=temp.tolist(),
            y=tx[tx["product_category"] == "fan"].groupby(day)["quantity"].sum()
              .reindex(temp.index, fill_value=0).tolist(),
            mode="markers")),
    }
    for name, fig in ok.items():
        res = _plot_check("G5_05", fig, tmp_path, name)
        assert res.passed, (name, res.reason)
    for name, fig in bad.items():
        assert not _plot_check("G5_05", fig, tmp_path, name).passed, name


def test_plot_g5_06_sorted_bars(tmp_path):
    s = _series("G5_06")["expected_series"]
    desc = sorted(s.items(), key=lambda kv: -kv[1])
    asc = list(reversed(desc))
    ok = {"desc": go.Figure(go.Bar(x=[k for k, _ in desc], y=[v for _, v in desc]))}
    bad = {
        "asc": go.Figure(go.Bar(x=[k for k, _ in asc], y=[v for _, v in asc])),
        "alphabetical": go.Figure(go.Bar(x=list(s), y=list(s.values()))),
    }
    assert list(s) != [k for k, _ in desc]  # alphabetical order differs from desc, so the test means something
    for name, fig in ok.items():
        assert _plot_check("G5_06", fig, tmp_path, name).passed, name
    for name, fig in bad.items():
        assert not _plot_check("G5_06", fig, tmp_path, name).passed, name


def test_every_plot_task_has_a_figure_test():
    tested = {"G5_01", "G5_02", "G5_03", "G5_04", "G5_05", "G5_06"}
    assert {t["id"] for t in TASKS if t["checker"] == "plot"} <= tested


# --------------------------------------------------------------------------
# ChatGPT template
# --------------------------------------------------------------------------
def test_chatgpt_prompts():
    for t in TASKS:
        prompt = T.render_chatgpt_prompt(t)
        assert t["question"] in prompt
        if t["checker"] == "plot":
            assert "ANSWER_JSON" not in prompt and T.PLOT_SENTENCE in prompt
        else:
            assert "ANSWER_JSON:" in prompt and t["answer_format"] in prompt


def test_chatgpt_template_build_and_overwrite_guard(tmp_path):
    from openpyxl import load_workbook
    out = tmp_path / "chatgpt_results.xlsx"
    T.build(out)
    wb = load_workbook(out)
    assert wb.sheetnames == ["results", "checklist_agent", "instructions"]
    ws = wb["results"]
    header = [c.value for c in ws[1]]
    assert header == T.RESULT_COLS
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    assert len(rows) == 12
    assert sorted(r[1] for r in rows) == sorted(GROUPS * 2)
    # simulate a manual entry, then a re-run must refuse to overwrite
    ws.cell(row=2, column=header.index("answer_line") + 1, value="ANSWER_JSON: 1")
    wb.save(out)
    with pytest.raises(FileExistsError):
        T.build(out)
    T.build(out, force=True)
