"""GD2 unit tests for eval/checkers.py (each rule tested with right AND wrong inputs).

Run: .venv\\Scripts\\python.exe -m pytest tests\\test_checkers.py
"""
import base64

import numpy as np
import pytest

import checkers as C


# --------------------------------------------------------------------------
# ANSWER_JSON parsing (ChatGPT line + FINAL fallback)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,found,value", [
    ("blah\nANSWER_JSON: 12.5", True, 12.5),
    ("ANSWER_JSON: {\"a\": 1, \"b\": [1, 2]}", True, {"a": 1, "b": [1, 2]}),
    ("answer_json: true", True, True),
    ("**ANSWER_JSON:** 25", True, 25),
    ("ANSWER_JSON: `{\"x\": 1}`", True, {"x": 1}),
    ("ANSWER_JSON: ```json {\"x\": 1}```", True, {"x": 1}),
    ("ANSWER_JSON: 7.", True, 7),
    ("ANSWER_JSON: 1\nmore text\nANSWER_JSON: 2", True, 2),   # last line wins
    ("ANSWER_JSON: {not json}", False, None),
    ("ANSWER_JSON: 1,5", False, None),                         # decimal comma is not JSON
    ("The answer is 12.5", False, None),
    ("", False, None),
    (None, False, None),
])
def test_parse_answer_text(text, found, value):
    assert C.parse_answer_text(text) == (found, value)


@pytest.mark.parametrize("text,checker,found,value,how", [
    ("Total revenue is 1234.56.", "number", True, 1234.56, "single_number"),
    ("Total revenue is 1,234.56 EUR", "number", True, 1234.56, "single_number"),
    ("There are 25 duplicate rows", "integer", True, 25.0, "single_number"),
    ("Between 2023 and 2024 revenue grew 11.2%", "number", False, None, "none"),  # 3 numbers
    ("Best month is 2024-12", "number", False, None, "none"),
    ("Region North is the best", "string", False, None, "none"),  # never guess non-numbers
    ("Done.\nANSWER_JSON: {\"x\": 3}", "composite", True, {"x": 3}, "answer_json_line"),
    (None, "number", False, None, "none"),
])
def test_extract_answer_from_final(text, checker, found, value, how):
    assert C.extract_answer_from_final(text, checker) == (found, value, how)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("x,expected", [
    (3, 3.0), (2.5, 2.5), (np.int64(4), 4.0), ("12.5", 12.5), ("74.2%", 74.2),
    ("1,234.5", 1234.5), (" 7 ", 7.0),
    (True, None), (None, None), ("abc", None), ("nan", None), (float("inf"), None), ([1], None),
])
def test_to_number(x, expected):
    assert C.to_number(x) == expected


# --------------------------------------------------------------------------
# Scalar checkers
# --------------------------------------------------------------------------
def _run(name, ans, exp, params=None, **ctx):
    return C.CHECKERS[name](ans, exp, params or {}, ctx)[0]


def test_number_tolerances():
    assert _run("number", 0.684, 0.68412, {"tol_abs": 0.005})
    assert not _run("number", 0.67, 0.68412, {"tol_abs": 0.005})
    assert _run("number", 1000.4, 1000.0, {"tol_rel": 0.0005})
    assert not _run("number", 1001.0, 1000.0, {"tol_rel": 0.0005})
    assert not _run("number", "abc", 1.0)
    assert not _run("number", True, 1.0)  # booleans are not numbers


def test_integer():
    assert _run("integer", 25, 25)
    assert _run("integer", 25.0, 25)
    assert _run("integer", "25", 25)
    assert not _run("integer", 25.5, 25)
    assert not _run("integer", 26, 25)


def test_string_and_aliases():
    assert _run("string", "  north ", "North")
    assert not _run("string", "South", "North")
    assert _run("string", "temperature", "temp_c", {"aliases": ["temperature"]})
    assert not _run("string", None, "North")
    assert not _run("string", "north", "North", {"case_sensitive": True})


def test_set_and_list():
    assert _run("set", ["email", "Total_Spent"], ["total_spent", "email"])
    assert not _run("set", ["email"], ["total_spent", "email"])
    assert not _run("set", ["email", "total_spent", "age"], ["total_spent", "email"])
    assert not _run("set", ["email", "email"], ["email"])  # duplicates are an error
    assert not _run("set", "email", ["email"])
    assert _run("list", ["a", "b"], ["A", "B"])
    assert not _run("list", ["b", "a"], ["a", "b"])


def test_bool():
    assert _run("bool", False, False)
    assert _run("bool", "false", False)
    assert not _run("bool", True, False)
    assert not _run("bool", "maybe", False)
    assert not _run("bool", 0.5, False)


def test_datetime():
    assert _run("datetime", "2024-03-01T00:00:00", "2024-03-01 00:00:00")
    assert not _run("datetime", "2024-03-01 01:00:00", "2024-03-01 00:00:00")
    assert not _run("datetime", "not a date", "2024-03-01")


def test_dict_number():
    exp = {"North": 100.0, "South": 50.0}
    assert _run("dict_number", {"north": 100.04, "SOUTH": 50}, exp, {"tol_rel": 0.001})
    assert not _run("dict_number", {"north": 101, "south": 50}, exp, {"tol_rel": 0.001})
    assert not _run("dict_number", {"north": 100}, exp)                       # missing key
    assert not _run("dict_number", {"north": 100, "south": 50, "x": 1}, exp)  # extra key
    assert _run("dict_number", {"north": 100, "south": 50, "x": 1}, exp, {"allow_extra_keys": True})
    assert not _run("dict_number", [100, 50], exp)


def test_composite_reports_each_field():
    p = {"fields": {"a": {"checker": "integer"}, "b": {"checker": "string"}}}
    ok, _, det = C.chk_composite({"A": 1, "b": "x"}, {"a": 1, "b": "x"}, p, {})
    assert ok
    ok, reason, det = C.chk_composite({"a": 1, "b": "y"}, {"a": 1, "b": "x"}, p, {})
    assert not ok and reason == "1/2 fields correct"
    assert det["fields"]["b"]["passed"] is False
    ok, _, det = C.chk_composite({"a": 1}, {"a": 1, "b": "x"}, p, {})
    assert not ok and det["fields"]["b"]["reason"] == "missing field"


# --------------------------------------------------------------------------
# Correlation is not causation
# --------------------------------------------------------------------------
X, Y = ["ice cream", "ice_cream"], ["fan"]
VX, VY = ["kem"], ["quạt"]  # Vietnamese sentences need Vietnamese terms

NOT_CLAIMS = [
    ("Daily ice cream and fan sales are strongly correlated (r = 0.68).", X, Y),
    ("Ice cream sales do not cause fan sales; both rise with temperature.", X, Y),
    ("Temperature drives both ice cream and fan sales.", X, Y),
    ("Correlation does not imply that ice cream drives fan sales.", X, Y),
    ("It is unlikely that ice cream causes fan purchases.", X, Y),
    ("Ice cream sales and fan sales both increase with temperature.", X, Y),
    ("Temperature, not ice cream, drives fan sales.", X, Y),
    ("Doanh số kem không gây ra doanh số quạt.", VX, VY),
    ("Nhiệt độ làm tăng cả doanh số kem và quạt.", VX, VY),
]
CLAIMS = [
    ("Ice cream sales cause fan sales.", X, Y),
    ("Higher ice cream sales lead to more fans being sold.", X, Y),
    ("Buying ice cream makes people buy a fan.", X, Y),
    ("Fan purchases drive ice cream sales.", X, Y),
    ("The data shows ice_cream demand increases fan demand.", X, Y),
    ("Correlation is 0.68. Therefore ice cream causes fan sales.", X, Y),
    ("Doanh số kem dẫn đến tăng doanh số quạt.", VX, VY),
    ("Bán kem gây ra việc bán quạt.", VX, VY),
]


@pytest.mark.parametrize("text,x,y", NOT_CLAIMS)
def test_no_causal_claim_accepts(text, x, y):
    ok, bad = C.no_causal_claim(text, x, y)
    assert ok, bad


@pytest.mark.parametrize("text,x,y", CLAIMS)
def test_no_causal_claim_rejects(text, x, y):
    ok, bad = C.no_causal_claim(text, x, y)
    assert not ok and bad


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def test_decode_array_lists_and_bdata():
    assert C.decode_array([1, 2]) == [1, 2]
    arr = np.array([1.5, 2.5, 3.5])
    enc = {"dtype": "f8", "bdata": base64.b64encode(arr.tobytes()).decode()}
    assert C.decode_array(enc) == [1.5, 2.5, 3.5]
    arr2 = np.arange(6, dtype="i4")
    enc2 = {"dtype": "i4", "bdata": base64.b64encode(arr2.tobytes()).decode(), "shape": "2, 3"}
    assert C.decode_array(enc2) == [[0, 1, 2], [3, 4, 5]]
    assert C.decode_array(None) is None


def test_trace_kinds_default_modes():
    assert C.trace_kinds({"type": "scatter", "x": list(range(5))}) == {"line", "scatter"}
    assert C.trace_kinds({"type": "scatter", "x": list(range(30))}) == {"line"}
    assert C.trace_kinds({"type": "scatter", "mode": "markers"}) == {"scatter"}
    assert C.trace_kinds({"type": "bar"}) == {"bar"}


def test_norm_x_month_and_day():
    assert C.norm_x(["2023-01", "2023-02-15", "2023-03-01T00:00:00"], "date_month") == \
        ["2023-01", "2023-02", "2023-03"]
    assert C.norm_x(["2024-03-01", "2024-03-02 00:00:00"], "date_day") == ["2024-03-01", "2024-03-02"]
    assert C.norm_x(["S1", "not a date"], "date_day") is None


# --------------------------------------------------------------------------
# check(): status values, never raises
# --------------------------------------------------------------------------
GT = {"datasets": {"d": {"facts": {"n": 5, "m": {"a": 1.0}}}}}


def test_check_statuses_and_gt_resolution():
    task = {"checker": "integer", "expected": {"gt": "d.facts.n"}, "params": {}}
    assert C.check(task, 5, GT).status == "PASS"
    assert C.check(task, 4, GT).status == "FAIL"
    assert C.check(task, C.NO_ANSWER, GT).status == "NO_ANSWER"


def test_check_bad_gt_path_raises_keyerror():
    task = {"checker": "integer", "expected": {"gt": "d.facts.missing"}, "params": {}}
    with pytest.raises(KeyError):
        C.check(task, 5, GT)


def test_check_never_raises_on_weird_answers():
    task = {"checker": "composite", "expected": {"value": {"m": {"gt": "d.facts.m"}}},
            "params": {"fields": {"m": {"checker": "dict_number"}}}}
    for weird in (None, 1, "x", [1], {"m": None}, {"m": [1, 2]}, {"m": {"a": "b"}}):
        assert C.check(task, weird, GT).status == "FAIL"
