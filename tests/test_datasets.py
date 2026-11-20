"""GD1 tests: synthetic datasets, planted traps and ground truth.

Run from the project root:
    .venv\\Scripts\\python.exe -m pytest tests\\test_datasets.py
"""
import json
from pathlib import Path

import pandas as pd
import pytest

import make_datasets as md

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_GT = ROOT / "datasets" / md.GT_FILE


@pytest.fixture(scope="session")
def gen(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    gt = md.build_all(out, md.DEFAULT_SEED)
    return out, gt


def _ds(gen, name):
    out, gt = gen
    d = gt["datasets"][name]
    return out / d["file"], d["injected"], d["facts"]


# --------------------------------------------------------------------------
# Generic properties
# --------------------------------------------------------------------------
def test_all_files_written_and_nonempty(gen):
    out, gt = gen
    assert len(gt["datasets"]) == 7
    for name, d in gt["datasets"].items():
        p = out / d["file"]
        assert p.exists() and p.stat().st_size > 0, name


def test_all_required_formats_present(gen):
    _, gt = gen
    assert {d["format"] for d in gt["datasets"].values()} == {"csv", "xlsx", "tsv", "json", "parquet"}


def test_ground_truth_is_strict_utf8_json(gen):
    out, _ = gen
    text = (out / md.GT_FILE).read_text(encoding="utf-8")

    def reject(c):
        raise ValueError(f"non-standard JSON constant {c}")

    json.loads(text, parse_constant=reject)  # NaN / Infinity would raise
    assert "\r\n" not in text


def test_generation_is_deterministic(gen, tmp_path):
    _, gt = gen
    assert md.build_all(tmp_path, md.DEFAULT_SEED) == gt


def test_seed_changes_data(gen, tmp_path):
    _, gt = gen
    other = md.build_all(tmp_path, md.DEFAULT_SEED + 1)
    a = gt["datasets"]["sales"]["facts"]["total_revenue_dedup"]
    b = other["datasets"]["sales"]["facts"]["total_revenue_dedup"]
    assert a != b


def test_committed_ground_truth_matches_generator(gen):
    assert COMMITTED_GT.exists(), "run: python datasets/make_datasets.py"
    committed = json.loads(COMMITTED_GT.read_text(encoding="utf-8"))
    assert committed == gen[1], "datasets/ is stale: re-run make_datasets.py"


def test_csv_files_use_lf_line_endings(gen):
    for name in ("sales", "sensors", "eu_sales"):
        path, _, _ = _ds(gen, name)
        assert b"\r\n" not in path.read_bytes(), name


# --------------------------------------------------------------------------
# Planted traps: injected (by construction) == facts (recomputed from file)
# --------------------------------------------------------------------------
def test_sales_traps(gen):
    path, inj, f = _ds(gen, "sales")
    assert f["n_rows_raw"] == inj["n_rows_raw"] == 2025
    assert f["n_exact_duplicate_rows"] == inj["n_exact_duplicate_rows"] == 25
    assert f["n_rows_dedup"] == inj["n_unique_orders"]
    assert f["missing_dedup"]["region"] == inj["n_missing_region_unique_rows"]
    assert f["missing_dedup"]["discount"] == inj["n_missing_discount_unique_rows"]
    assert f["n_quantity_ge_300_dedup"] == inj["n_bulk_orders_qty_ge_300"]
    assert f["quantity_sum_by_product_dedup"]["Notebook"] > f["quantity_sum_by_product_dedup"]["Laptop"]
    assert 0 < f["yoy_revenue_growth_pct_2024_dedup"] < 40  # planted growth, not outlier-driven
    assert f["total_revenue_raw"] > f["total_revenue_dedup"]  # duplicates inflate totals
    assert f["best_month_by_revenue_dedup"] == "2024-12"  # planted Q4 seasonality


def test_hr_traps(gen):
    path, inj, f = _ds(gen, "hr")
    assert f["sheet_names"] == inj["sheet_names"]
    assert f["first_sheet"] == "README"
    naive = pd.read_excel(path)  # default = first sheet = notes, not data
    assert list(naive.columns) == ["note"]
    assert f["n_orphan_employees"] == inj["n_orphan_employees"] == 5
    assert f["orphan_dept_ids"] == inj["orphan_dept_ids"] == [99]
    assert f["n_employees_after_inner_join"] == inj["n_employees"] - inj["n_orphan_employees"]
    assert f["hire_date_dtype"].startswith("datetime64")


def test_sensors_traps(gen):
    path, inj, f = _ds(gen, "sensors")
    assert f["is_sorted_by_timestamp_in_file"] is False
    assert f["expected_rows_per_sensor"] == inj["expected_rows_per_sensor"] == 744
    assert f["missing_hours_per_sensor"] == inj["missing_hours_per_sensor"]
    assert f["longest_gap_hours_per_sensor"]["S2"] == inj["s2_outage_hours"]
    assert f["longest_constant_temp_run_per_sensor"]["S1"] == inj["s1_frozen_hours"]
    assert f["longest_constant_temp_run_start_per_sensor"]["S1"] == inj["s1_frozen_start"]
    assert f["n_missing_humidity"] == inj["n_missing_humidity"]
    assert f["n_rows"] == 3 * 744 - 12 - 20


def test_customers_traps(gen):
    path, inj, f = _ds(gen, "customers")
    text = path.read_text(encoding="utf-8")
    assert "Hà Nội" in text and "Nguyễn" in text  # stored as real UTF-8, not \u escapes
    assert f["n_rows"] == inj["n_rows"]
    assert f["n_age_stored_as_string"] == inj["n_age_stored_as_string"] == 30
    assert f["missing"]["total_spent"] == inj["n_missing_total_spent"]
    assert f["missing"]["email"] == inj["n_missing_email"]


def test_eu_sales_traps(gen):
    path, inj, f = _ds(gen, "eu_sales")
    assert f["n_cols"] == inj["n_cols"] == 6
    assert f["dtypes"]["revenue_eur"] == "float64"
    naive = f["naive_read_csv_default"]  # default read_csv must NOT give 6 columns
    assert (not naive["ok"]) or naive["n_cols"] != 6


def test_clinic_traps(gen):
    path, inj, f = _ds(gen, "clinic")
    assert f["visit_date_dtype_raw"] == "object"
    for k in ("n_visit_date_missing", "n_visit_date_invalid", "n_visit_date_valid",
              "n_visit_date_valid_text", "n_age_missing", "n_age_invalid",
              "n_age_numeric_text", "n_bp_missing", "n_bp_invalid", "n_cost_text",
              "total_cost_vnd"):
        assert f[k] == inj[k], k
    assert f["mean_age_valid"] == pytest.approx(inj["mean_age_valid"], abs=1e-6)
    assert f["n_sex_missing"] == inj["n_sex_missing"]
    assert f["sex_counts_normalized"]["unknown"] == inj["n_sex_missing"]
    # day-first parsing is verified against the true dates
    assert f["visits_per_month_valid"] == inj["visits_per_month_valid"]


def test_clinic_garbage_tokens_survive_pandas_na_detection():
    # Regression: "N/A" was read as NaN by read_excel, turning "invalid" into "missing".
    default_na = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
                  "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
                  "n/a", "nan", "null"}
    assert not default_na & set(md.CLINIC_GARBAGE)


def test_transactions_confounder(gen):
    path, inj, f = _ds(gen, "transactions")
    assert f["n_rows"] == inj["n_rows"] > 30000
    for c in ("store_id", "product_category", "payment_method"):
        assert f["dtypes"][c] == "category"
    assert f["dtypes"]["ts"].startswith("datetime64")
    assert f["corr_daily_units_icecream_fan"] > 0.5            # strong raw correlation
    assert abs(f["partial_corr_icecream_fan_given_temp"]) < 0.15  # vanishes given temp
    assert f["corr_daily_units_icecream_temp"] > 0.7
    assert abs(f["corr_daily_units_bread_icecream"]) < 0.2       # control category


# --------------------------------------------------------------------------
# Canonical parsing rules (unit level)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value,status,iso", [
    ("2024-03-05", "valid", "2024-03-05"),
    ("05/03/2024", "valid", "2024-03-05"),   # day first
    ("March 5, 2024", "valid", "2024-03-05"),
    (pd.Timestamp("2024-03-05"), "valid", "2024-03-05"),
    ("2024-02-30", "invalid", None),
    ("29/02/2023", "invalid", None),          # 2023 is not a leap year
    ("29/02/2024", "valid", "2024-02-29"),
    ("Smarch 5, 2024", "invalid", None),
    ("unknown", "invalid", None),
    (None, "missing", None),
    (float("nan"), "missing", None),
    ("   ", "missing", None),
])
def test_parse_clinic_date(value, status, iso):
    st, ts = md.parse_clinic_date(value)
    assert st == status
    assert (ts.strftime("%Y-%m-%d") if ts is not None else None) == iso


@pytest.mark.parametrize("value,expected", [
    ("M", "M"), ("male", "M"), ("Female", "F"), ("f", "F"), (None, None), ("x", None),
])
def test_normalize_sex(value, expected):
    assert md.normalize_sex(value) == expected


@pytest.mark.parametrize("value,expected", [
    (1250000, 1250000), ("1.250.000", 1250000), ("abc", None), (None, None),
])
def test_parse_cost_vnd(value, expected):
    assert md.parse_cost_vnd(value) == expected


# --------------------------------------------------------------------------
# Real datasets (only if downloaded with datasets/fetch_real.py)
# --------------------------------------------------------------------------
REAL_SOURCES = ROOT / "datasets" / "real" / "SOURCES.json"


@pytest.mark.skipif(not REAL_SOURCES.exists(), reason="real datasets not downloaded yet")
def test_real_datasets_match_recorded_checksums():
    import fetch_real
    assert fetch_real.verify()
