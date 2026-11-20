"""GD3: data.py - one loader for every format, sheet choice, traps, profiling."""
import json
from pathlib import Path

import pandas as pd
import pytest

from data import DataLoadError, list_sheets, load_data, profile, profile_file, sniff_csv

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "datasets" / "synthetic"
REAL = ROOT / "datasets" / "real"
GT = json.loads((ROOT / "datasets" / "ground_truth.json").read_text(encoding="utf-8"))["datasets"]


# --------------------------------------------------------------------------
# Every project dataset loads with the right shape (ground truth)
# --------------------------------------------------------------------------
def test_sales_csv():
    df = load_data(SYN / "sales.csv")
    assert df.shape == (GT["sales"]["facts"]["n_rows_raw"], GT["sales"]["facts"]["n_cols"])
    assert df.attrs["source"]["sep"] == ","


def test_eu_sales_semicolon_decimal_comma():
    df = load_data(SYN / "eu_sales.csv")
    assert df.attrs["source"]["sep"] == ";" and df.attrs["source"]["decimal"] == ","
    assert list(df.columns) == GT["eu_sales"]["facts"]["columns"]
    assert str(df["revenue_eur"].dtype) == "float64"
    ref = pd.read_csv(SYN / "eu_sales.csv", sep=";", decimal=",")
    pd.testing.assert_frame_equal(df, ref)
    assert df["revenue_eur"].sum() == pytest.approx(GT["eu_sales"]["facts"]["total_revenue_eur"], rel=1e-9)


def test_sensors_tsv():
    df = load_data(SYN / "sensors.tsv")
    assert len(df) == GT["sensors"]["facts"]["n_rows"] and df.attrs["source"]["sep"] == "\t"


def test_hr_default_sheet_is_largest_not_first():
    df = load_data(SYN / "hr.xlsx")
    assert df.attrs["source"]["sheet"] == "employees"
    assert len(df) == GT["hr"]["facts"]["n_employees"]
    assert "sheets" not in df.attrs["source"]  # the loader never lists the other sheets


def test_sheet_policy_first_behaves_like_pandas(monkeypatch):
    df = load_data(SYN / "hr.xlsx", sheet_policy="first")
    assert df.attrs["source"]["sheet"] == "README"
    pd.testing.assert_frame_equal(df, pd.read_excel(SYN / "hr.xlsx"))
    monkeypatch.setenv("EDA_DEFAULT_SHEET", "first")   # how the sandbox sets it
    assert load_data(SYN / "hr.xlsx").attrs["source"]["sheet"] == "README"
    assert load_data(SYN / "hr.xlsx", sheet_policy="largest").attrs["source"]["sheet"] == "employees"
    monkeypatch.setenv("EDA_DEFAULT_SHEET", "biggest")
    with pytest.raises(DataLoadError, match="EDA_DEFAULT_SHEET"):
        load_data(SYN / "hr.xlsx")


def test_hr_explicit_sheets():
    assert list(load_data(SYN / "hr.xlsx", sheet="README").columns) == ["note"]
    assert list(load_data(SYN / "hr.xlsx", sheet=0).columns) == ["note"]
    assert len(load_data(SYN / "hr.xlsx", sheet="departments")) == GT["hr"]["facts"]["n_departments"]
    with pytest.raises(DataLoadError, match="no sheet named 'employes'.*employees"):
        load_data(SYN / "hr.xlsx", sheet="employes")
    with pytest.raises(DataLoadError, match="out of range"):
        load_data(SYN / "hr.xlsx", sheet=7)


def test_list_sheets():
    sheets = list_sheets(SYN / "clinic_messy.xlsx")
    assert [s["name"] for s in sheets] == ["patients", "data_dictionary"]
    assert sheets[0]["rows"] == 400
    assert list_sheets(SYN / "sales.csv") == []


def test_clinic_keeps_mixed_types():
    df = load_data(SYN / "clinic_messy.xlsx")
    assert df.attrs["source"]["sheet"] == "patients"
    assert str(df["visit_date"].dtype) == "object"


def test_customers_json_keeps_raw_types():
    df = load_data(SYN / "customers.json")
    assert len(df) == GT["customers"]["facts"]["n_rows"]
    n_str = sum(isinstance(v, str) for v in df["age"])
    assert n_str == GT["customers"]["facts"]["n_age_stored_as_string"]  # pd.read_json would give 0
    assert "Hà Nội" in set(df["city"])


def test_transactions_parquet():
    df = load_data(SYN / "transactions.parquet")
    assert len(df) == GT["transactions"]["facts"]["n_rows"]
    assert str(df["store_id"].dtype) == "category"


@pytest.mark.parametrize("name,rows", [("titanic.csv", 891), ("penguins.csv", 344),
                                       ("tips.csv", 244), ("california_housing.csv", 20640)])
def test_real_datasets(name, rows):
    assert len(load_data(REAL / name)) == rows


# --------------------------------------------------------------------------
# Generic formats and encodings
# --------------------------------------------------------------------------
def test_csv_variants(tmp_path):
    (tmp_path / "pipe.csv").write_text("a|b\n1|x\n2|y\n", encoding="utf-8")
    assert list(load_data(tmp_path / "pipe.csv").columns) == ["a", "b"]

    (tmp_path / "bom.csv").write_text("name,v\nAn,1\n", encoding="utf-8-sig")
    assert list(load_data(tmp_path / "bom.csv").columns) == ["name", "v"]  # no BOM in header

    (tmp_path / "latin.csv").write_bytes("city;price\nMünchen;1,5\nKöln;2,25\n".encode("cp1252"))
    df = load_data(tmp_path / "latin.csv")
    assert df["city"].tolist() == ["München", "Köln"] and df["price"].tolist() == [1.5, 2.25]
    assert df.attrs["source"]["encoding"] == "cp1252"

    (tmp_path / "data.txt").write_text("x\ty\n1\t2\n", encoding="utf-8")
    assert df.shape[1] == 2 and load_data(tmp_path / "data.txt").shape == (1, 2)

    (tmp_path / "vn.csv").write_text("tinh,so\nHà Nội,1\nĐà Nẵng,2\n", encoding="utf-8")
    assert load_data(tmp_path / "vn.csv")["tinh"].tolist() == ["Hà Nội", "Đà Nẵng"]


def test_sniff_csv():
    assert sniff_csv("a,b,c\n1,2,3\n") == (",", ".")
    assert sniff_csv("a;b\n1,5;2\n3,25;4\n") == (";", ",")
    assert sniff_csv("a;b\n1.5;2\n") == (";", ".")
    assert sniff_csv("") == (",", ".")


def test_json_layouts(tmp_path):
    (tmp_path / "cols.json").write_text(json.dumps({"a": [1, 2], "b": ["x", "y"]}), encoding="utf-8")
    assert load_data(tmp_path / "cols.json").shape == (2, 2)
    (tmp_path / "wrapped.json").write_text(json.dumps({"data": [{"a": 1}, {"a": 2}]}), encoding="utf-8")
    assert load_data(tmp_path / "wrapped.json")["a"].tolist() == [1, 2]
    (tmp_path / "lines.jsonl").write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert load_data(tmp_path / "lines.jsonl")["a"].tolist() == [1, 2]
    (tmp_path / "bad.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(DataLoadError, match="unsupported JSON layout"):
        load_data(tmp_path / "bad.json")


def test_unsupported_and_missing(tmp_path):
    (tmp_path / "x.xls").write_bytes(b"")
    (tmp_path / "x.docx").write_bytes(b"")
    with pytest.raises(DataLoadError, match="xlsx"):
        load_data(tmp_path / "x.xls")
    with pytest.raises(DataLoadError, match="Unsupported"):
        load_data(tmp_path / "x.docx")
    with pytest.raises(DataLoadError, match="not found"):
        load_data(tmp_path / "nope.csv")


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------
def test_profile_sections_and_hints():
    p = profile(load_data(SYN / "customers.json"))
    assert p.startswith("SHAPE: 500 rows x 10 columns")
    for section in ("COLUMNS", "NUMERIC SUMMARY", "CATEGORICAL TOP", "TOP PEARSON CORRELATIONS"):
        assert section in p
    assert "mixed types: int=470, str=30" in p
    assert "correlation is not causation" in p


def test_profile_detects_text_dates():
    p = profile(load_data(SYN / "clinic_messy.xlsx"))
    line = next(ln for ln in p.splitlines() if ln.startswith("- visit_date |"))
    assert "mixed types: datetime=276, str=112" in line and "text that looks like dates" in line
    age = next(ln for ln in p.splitlines() if ln.startswith("- age |"))
    assert "mixed types: int=383, str=14" in age and "looks like dates" not in age


def test_profile_is_deterministic_and_bounded():
    df = load_data(SYN / "transactions.parquet")
    assert profile(df) == profile(df)
    short = profile(df, max_chars=600)
    assert len(short) <= 600 and short.endswith("[profile truncated]")


def test_profile_file_lists_sheets():
    p = profile_file(SYN / "hr.xlsx", max_chars=3000)
    assert p.splitlines()[0].startswith("SHEETS: README (5 rows x 1 cols); employees (300 rows")
    assert "PROFILED SHEET: employees (load it with load_data(path, sheet='employees'))" in p
    assert len(p) <= 3000
