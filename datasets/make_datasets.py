"""Generate synthetic EDA datasets with planted traps + ground truth.

Usage (from the project root, Windows):
    .venv\\Scripts\\python.exe datasets\\make_datasets.py
    .venv\\Scripts\\python.exe datasets\\make_datasets.py --out some\\dir --seed 42

How ground truth is built (two independent paths, cross-checked by tests):
  1. make_<name>(rng, path): builds the data from a seeded RNG, writes the file,
     and returns "injected" = what was planted, known by construction.
  2. facts_<name>(path): READS THE FILE BACK from disk with the correct
     parameters and computes "facts" with a canonical pandas procedure.
  tests/test_datasets.py asserts injected == facts for every planted trap.

Output:
  <out>/synthetic/*.csv|xlsx|tsv|json|parquet
  <out>/ground_truth.json   {meta, datasets: {name: {file, format, read_hint,
                                                      traps, injected, facts}}}

Every dataset gets its own RNG stream (seed, dataset index), so adding or
reordering datasets never changes the others. No wall-clock time is used
anywhere: "today" is the fixed REFERENCE_DATE.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

GT_VERSION = "1.0"
DEFAULT_SEED = 42
REFERENCE_DATE = "2025-01-01"
FLOAT_DIGITS = 6
SYNTH_DIR = "synthetic"
GT_FILE = "ground_truth.json"

# English month names are hard-coded: datetime.strptime("%B") depends on the
# process locale, which differs between Windows (vi-VN) and Kaggle (C/POSIX).
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]

VN_FAMILY = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ",
             "Võ", "Đặng", "Bùi", "Đỗ"]
VN_MIDDLE = ["Văn", "Thị", "Minh", "Ngọc", "Đức", "Thu", "Quốc", "Thanh"]
VN_GIVEN = ["An", "Bình", "Châu", "Dũng", "Hà", "Hương", "Khoa", "Lan", "Long",
            "Mai", "Nam", "Phúc", "Quân", "Thảo", "Trang", "Tú", "Việt", "Yến"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _clean(obj):
    """Convert numpy/pandas scalars to strict-JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, FLOAT_DIGITS)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def _write_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    # lineterminator="\n": pandas would otherwise write "\r\n" on Windows and
    # "\n" on Linux/Kaggle, giving different files from the same seed.
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", **kwargs)


def _vc(series: pd.Series) -> dict:
    """value_counts as {str(key): int}, sorted by key, NaN excluded."""
    return {str(k): int(v) for k, v in series.value_counts(dropna=True).sort_index().items()}


def _series_dict(series: pd.Series) -> dict:
    return {str(k): v for k, v in series.sort_index().items()}


def _vn_names(rng: np.random.Generator, n: int) -> list[str]:
    fam = rng.choice(VN_FAMILY, size=n)
    mid = rng.choice(VN_MIDDLE, size=n)
    giv = rng.choice(VN_GIVEN, size=n)
    return [f"{a} {b} {c}" for a, b, c in zip(fam, mid, giv)]


def _disjoint(rng: np.random.Generator, n: int, sizes: dict[str, int]) -> dict[str, np.ndarray]:
    """Pick disjoint random index sets of the given sizes from range(n)."""
    if sum(sizes.values()) > n:
        raise ValueError("sizes exceed population")
    perm = rng.permutation(n)
    out, pos = {}, 0
    for key, size in sizes.items():
        out[key] = perm[pos:pos + size]
        pos += size
    return out


def _longest_true_run(mask) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _longest_equal_run(values) -> tuple[int, int]:
    """Length and start index of the longest run of identical consecutive values."""
    best_len, best_start, cur_len, cur_start = 0, 0, 0, 0
    prev = object()
    for i, v in enumerate(values):
        if cur_len and v == prev:
            cur_len += 1
        else:
            cur_len, cur_start = 1, i
        prev = v
        if cur_len > best_len:
            best_len, best_start = cur_len, cur_start
    return best_len, best_start


def _iqr_outliers(s: pd.Series) -> tuple[float, float, int]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return float(lo), float(hi), int(((s < lo) | (s > hi)).sum())


def _is_blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------
# 1) sales.csv - orders with duplicates, missing values, bulk-order outliers
# ----------------------------------------------------------------------------
SALES_REGIONS = ["North", "South", "East", "West", "Central"]
SALES_REGION_P = [0.25, 0.20, 0.20, 0.15, 0.20]
# name, category, base unit price, poisson lambda for quantity
SALES_PRODUCTS = [
    ("Laptop", "Electronics", 850.0, 1.2),
    ("Phone", "Electronics", 520.0, 1.5),
    ("Headphones", "Electronics", 75.0, 2.0),
    ("Desk", "Furniture", 240.0, 1.0),
    ("Chair", "Furniture", 130.0, 2.5),
    ("Notebook", "Stationery", 4.5, 8.0),
    ("Pen Pack", "Stationery", 6.0, 6.0),
]
SALES_PRODUCT_P = [0.12, 0.15, 0.18, 0.08, 0.14, 0.18, 0.15]
SALES_BULK_PRODUCT_IDX = [5, 6]  # Notebook, Pen Pack
SALES_MONTH_W = [0.80, 0.75, 0.90, 0.90, 0.95, 1.00, 0.95, 0.95, 1.00, 1.10, 1.35, 1.50]
N_SALES = 2000
N_SALES_DUP = 25
N_SALES_BULK = 8
N_SALES_MISS_REGION = 40
N_SALES_MISS_DISCOUNT = 60


def make_sales(rng: np.random.Generator, path: Path) -> dict:
    n = N_SALES
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    w = np.array([SALES_MONTH_W[d.month - 1] * (1.10 if d.year == 2024 else 1.0) for d in days])
    w = w / w.sum()
    dates = pd.DatetimeIndex(np.sort(rng.choice(days.values, size=n, p=w)))

    pidx = rng.choice(len(SALES_PRODUCTS), size=n, p=SALES_PRODUCT_P)
    # Bulk orders are stationery only: a bulk laptop order (500 x 850 USD) would
    # dominate every total and hide the planted seasonality.
    bulk_idx = rng.choice(n, size=N_SALES_BULK, replace=False)
    pidx[bulk_idx] = rng.choice(SALES_BULK_PRODUCT_IDX, size=N_SALES_BULK)
    names = np.array([p[0] for p in SALES_PRODUCTS])[pidx]
    cats = np.array([p[1] for p in SALES_PRODUCTS])[pidx]
    base = np.array([p[2] for p in SALES_PRODUCTS])[pidx]
    lam = np.array([p[3] for p in SALES_PRODUCTS])[pidx]

    qty = rng.poisson(lam) + 1
    qty[bulk_idx] = rng.integers(300, 601, size=N_SALES_BULK)

    price_factor = np.where(dates.year == 2024, 1.05, 1.0)
    unit_price = np.round(base * price_factor * (1 + rng.normal(0, 0.02, size=n)), 2)
    discount = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    revenue = np.round(qty * unit_price * (1 - discount), 2)
    region = rng.choice(SALES_REGIONS, size=n, p=SALES_REGION_P).astype(object)

    df = pd.DataFrame({
        "order_id": np.arange(100001, 100001 + n),
        "order_date": dates.strftime("%Y-%m-%d"),
        "region": region,
        "product": names,
        "category": cats,
        "quantity": qty,
        "unit_price": unit_price,
        "discount": discount,
        "revenue": revenue,
    })
    # Missing values are planted in unique rows BEFORE duplication.
    df.loc[rng.choice(n, N_SALES_MISS_REGION, replace=False), "region"] = None
    df.loc[rng.choice(n, N_SALES_MISS_DISCOUNT, replace=False), "discount"] = np.nan

    dup_src = rng.choice(n, N_SALES_DUP, replace=False)
    out = (pd.concat([df, df.iloc[dup_src]], ignore_index=True)
           .sort_values(["order_date", "order_id"], kind="mergesort")
           .reset_index(drop=True))
    _write_csv(out, path)
    return {
        "n_unique_orders": n,
        "n_exact_duplicate_rows": N_SALES_DUP,
        "n_rows_raw": n + N_SALES_DUP,
        "n_bulk_orders_qty_ge_300": N_SALES_BULK,
        "n_missing_region_unique_rows": N_SALES_MISS_REGION,
        "n_missing_discount_unique_rows": N_SALES_MISS_DISCOUNT,
    }


def facts_sales(path: Path) -> dict:
    raw = pd.read_csv(path, encoding="utf-8")
    df = raw.drop_duplicates().reset_index(drop=True)
    d = pd.to_datetime(df["order_date"], format="%Y-%m-%d")
    f: dict = {
        "n_rows_raw": len(raw),
        "n_cols": raw.shape[1],
        "columns": list(raw.columns),
        "dtypes_raw": {c: str(t) for c, t in raw.dtypes.items()},
        "n_exact_duplicate_rows": int(raw.duplicated().sum()),
        "n_duplicate_order_id": int(raw["order_id"].duplicated().sum()),
        "n_rows_dedup": len(df),
        "missing_raw": raw.isna().sum().to_dict(),
        "missing_dedup": df.isna().sum().to_dict(),
        "date_min": d.min().strftime("%Y-%m-%d"),
        "date_max": d.max().strftime("%Y-%m-%d"),
        "total_revenue_raw": raw["revenue"].sum(),
        "total_revenue_dedup": df["revenue"].sum(),
        "mean_revenue_dedup": df["revenue"].mean(),
        "median_revenue_dedup": df["revenue"].median(),
    }
    reg = df.dropna(subset=["region"]).groupby("region")["revenue"]
    f["revenue_sum_by_region_dedup"] = _series_dict(reg.sum())
    f["revenue_mean_by_region_dedup"] = _series_dict(reg.mean())
    f["order_count_by_region_dedup"] = _series_dict(reg.size())
    f["top_region_by_revenue_dedup"] = str(reg.sum().idxmax())

    cat = df.groupby("category")["revenue"].sum()
    f["revenue_share_pct_by_category_dedup"] = _series_dict(cat / cat.sum() * 100)
    qty = df.groupby("product")["quantity"].sum()
    f["quantity_sum_by_product_dedup"] = _series_dict(qty)
    f["top_product_by_quantity_dedup"] = str(qty.idxmax())

    f["mean_discount_dedup_skipna"] = df["discount"].mean()
    f["share_orders_with_discount_pct_dedup"] = (df["discount"].dropna() > 0).mean() * 100

    monthly = df.groupby(d.dt.strftime("%Y-%m"))["revenue"].sum()
    f["monthly_revenue_dedup"] = _series_dict(monthly)
    f["best_month_by_revenue_dedup"] = str(monthly.idxmax())
    yearly = df.groupby(d.dt.year)["revenue"].sum()
    f["revenue_by_year_dedup"] = _series_dict(yearly)
    f["yoy_revenue_growth_pct_2024_dedup"] = (yearly.loc[2024] / yearly.loc[2023] - 1) * 100

    f["corr_quantity_revenue_pearson_dedup"] = df["quantity"].corr(df["revenue"])
    f["corr_unit_price_revenue_pearson_dedup"] = df["unit_price"].corr(df["revenue"])
    lo, hi, n_out = _iqr_outliers(df["revenue"])
    f["revenue_iqr_bounds_dedup"] = [lo, hi]
    f["n_revenue_outliers_iqr_dedup"] = n_out
    f["n_quantity_ge_300_dedup"] = int((df["quantity"] >= 300).sum())
    f["revenue_skew_dedup"] = df["revenue"].skew()
    return f


# ----------------------------------------------------------------------------
# 2) hr.xlsx - multi-sheet; data is NOT on the first sheet; orphan foreign keys
# ----------------------------------------------------------------------------
# dept_id, name, location, annual budget (USD), base monthly salary (USD)
HR_DEPTS = [
    (10, "Engineering", "Hanoi", 1_200_000, 2600),
    (20, "Sales", "HCMC", 800_000, 1800),
    (30, "Marketing", "HCMC", 500_000, 1700),
    (40, "Finance", "Hanoi", 400_000, 2200),
    (50, "HR", "Da Nang", 250_000, 1500),
    (60, "Operations", "Da Nang", 600_000, 1600),
]
HR_DEPT_P = [0.30, 0.22, 0.12, 0.10, 0.08, 0.18]
N_HR = 300
N_HR_ORPHAN = 5
HR_ORPHAN_DEPT = 99


def make_hr(rng: np.random.Generator, path: Path) -> dict:
    n = N_HR
    di = rng.choice(len(HR_DEPTS), size=n, p=HR_DEPT_P)
    dept_id = np.array([d[0] for d in HR_DEPTS])[di]
    base = np.array([d[4] for d in HR_DEPTS], dtype=float)[di]

    start, end = pd.Timestamp("2012-01-01"), pd.Timestamp("2024-06-30")
    hire = start + pd.to_timedelta(rng.integers(0, (end - start).days + 1, size=n), unit="D")
    tenure = np.asarray((pd.Timestamp(REFERENCE_DATE) - hire).days) / 365.25
    perf = np.clip(np.round(rng.normal(3.2, 0.9, size=n)), 1, 5).astype(int)
    salary = np.round(base * (1 + 0.03 * tenure) * (1 + 0.04 * (perf - 3))
                      * (1 + rng.normal(0, 0.08, size=n)), -1).astype(int)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.52, 0.48])
    remote = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    orphan = rng.choice(n, N_HR_ORPHAN, replace=False)
    dept_id = dept_id.copy()
    dept_id[orphan] = HR_ORPHAN_DEPT

    emp = pd.DataFrame({
        "emp_id": [f"E{i:04d}" for i in range(1, n + 1)],
        "full_name": _vn_names(rng, n),
        "dept_id": dept_id,
        "hire_date": hire,
        "gender": gender,
        "performance_score": perf,
        "monthly_salary_usd": salary,
        "remote": remote,
    })
    dept = pd.DataFrame(
        [(d[0], d[1], d[2], d[3]) for d in HR_DEPTS],
        columns=["dept_id", "dept_name", "location", "annual_budget_usd"],
    )
    readme = pd.DataFrame({"note": [
        "HR extract (synthetic data).",
        "Sheet 'employees': one row per employee.",
        "Sheet 'departments': department master data; join on dept_id.",
        "monthly_salary_usd is the monthly salary in USD.",
        f"Use {REFERENCE_DATE} as the reference date for tenure.",
    ]})
    with pd.ExcelWriter(path, engine="openpyxl", date_format="YYYY-MM-DD",
                        datetime_format="YYYY-MM-DD") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        emp.to_excel(xw, sheet_name="employees", index=False)
        dept.to_excel(xw, sheet_name="departments", index=False)
    return {
        "first_sheet": "README",
        "sheet_names": ["README", "employees", "departments"],
        "n_employees": n,
        "n_departments": len(HR_DEPTS),
        "n_orphan_employees": N_HR_ORPHAN,
        "orphan_dept_ids": [HR_ORPHAN_DEPT],
    }


def facts_hr(path: Path) -> dict:
    sheets = pd.ExcelFile(path, engine="openpyxl").sheet_names
    emp = pd.read_excel(path, sheet_name="employees", engine="openpyxl")
    dept = pd.read_excel(path, sheet_name="departments", engine="openpyxl")
    m = emp.merge(dept, on="dept_id", how="inner")
    orphan = ~emp["dept_id"].isin(dept["dept_id"])
    sal_dept = m.groupby("dept_name")["monthly_salary_usd"].mean()
    tenure = (pd.Timestamp(REFERENCE_DATE) - emp["hire_date"]).dt.days / 365.25
    return {
        "sheet_names": sheets,
        "first_sheet": sheets[0],
        "n_employees": len(emp),
        "n_departments": len(dept),
        "employee_columns": list(emp.columns),
        "department_columns": list(dept.columns),
        "hire_date_dtype": str(emp["hire_date"].dtype),
        "n_orphan_employees": int(orphan.sum()),
        "orphan_dept_ids": sorted(int(x) for x in emp.loc[orphan, "dept_id"].unique()),
        "n_employees_after_inner_join": len(m),
        "mean_salary_by_dept_name": _series_dict(sal_dept),
        "top_dept_by_mean_salary": str(sal_dept.idxmax()),
        "top_dept_mean_salary": sal_dept.max(),
        "headcount_by_dept_name": _series_dict(m.groupby("dept_name").size()),
        "headcount_by_location": _series_dict(m.groupby("location").size()),
        "gender_counts": _vc(emp["gender"]),
        "female_share_pct": (emp["gender"] == "Female").mean() * 100,
        "mean_salary_by_gender": _series_dict(emp.groupby("gender")["monthly_salary_usd"].mean()),
        "mean_salary_all": emp["monthly_salary_usd"].mean(),
        "median_salary_all": emp["monthly_salary_usd"].median(),
        "corr_performance_salary_pearson": emp["performance_score"].corr(emp["monthly_salary_usd"]),
        "median_tenure_years_at_reference_date": tenure.median(),
        "remote_share_pct": (emp["remote"] == "Yes").mean() * 100,
        "hires_by_year": _series_dict(emp.groupby(emp["hire_date"].dt.year).size()),
    }


# ----------------------------------------------------------------------------
# 3) sensors.tsv - hourly time series, unsorted rows, gaps, frozen sensor
# ----------------------------------------------------------------------------
SENSOR_BASE = {"S1": 24.0, "S2": 27.0, "S3": 21.0}
SENSOR_START = "2024-03-01 00:00:00"
SENSOR_HOURS = 31 * 24
S2_OUTAGE_START, S2_OUTAGE_HOURS = "2024-03-10 08:00:00", 12
S1_FROZEN_START, S1_FROZEN_HOURS = "2024-03-15 00:00:00", 10
N_S3_RANDOM_MISSING = 20
N_HUMIDITY_MISSING = 30
TS_FMT = "%Y-%m-%d %H:%M:%S"


def make_sensors(rng: np.random.Generator, path: Path) -> dict:
    ts = pd.date_range(SENSOR_START, periods=SENSOR_HOURS, freq="h")
    hours = ts.hour.values
    trend = np.linspace(0, 1.5, len(ts))
    frames = []
    for sid, base in SENSOR_BASE.items():
        temp = base + 4 * np.sin(2 * np.pi * (hours - 8) / 24) + trend + rng.normal(0, 0.4, len(ts))
        hum = np.clip(85 - 1.5 * (temp - base) - 2 * trend + rng.normal(0, 3, len(ts)), 30, 100)
        frames.append(pd.DataFrame({
            "timestamp": ts, "sensor_id": sid,
            "temperature_c": np.round(temp, 2), "humidity_pct": np.round(hum, 1),
        }))
    df = pd.concat(frames, ignore_index=True)

    frozen = pd.date_range(S1_FROZEN_START, periods=S1_FROZEN_HOURS, freq="h")
    is_s1 = df["sensor_id"] == "S1"
    frozen_val = df.loc[is_s1 & (df["timestamp"] == frozen[0]), "temperature_c"].iloc[0]
    df.loc[is_s1 & df["timestamp"].isin(frozen), "temperature_c"] = frozen_val

    outage = pd.date_range(S2_OUTAGE_START, periods=S2_OUTAGE_HOURS, freq="h")
    df = df[~((df["sensor_id"] == "S2") & df["timestamp"].isin(outage))]
    cand = ts[24:-24]
    drop_ts = cand[rng.choice(len(cand), N_S3_RANDOM_MISSING, replace=False)]
    df = df[~((df["sensor_id"] == "S3") & df["timestamp"].isin(drop_ts))].reset_index(drop=True)

    df.loc[rng.choice(len(df), N_HUMIDITY_MISSING, replace=False), "humidity_pct"] = np.nan
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime(TS_FMT)
    _write_csv(df, path, sep="\t")
    return {
        "expected_rows_per_sensor": SENSOR_HOURS,
        "missing_hours_per_sensor": {"S1": 0, "S2": S2_OUTAGE_HOURS, "S3": N_S3_RANDOM_MISSING},
        "s2_outage_start": S2_OUTAGE_START,
        "s2_outage_hours": S2_OUTAGE_HOURS,
        "s1_frozen_start": S1_FROZEN_START,
        "s1_frozen_hours": S1_FROZEN_HOURS,
        "n_missing_humidity": N_HUMIDITY_MISSING,
        "rows_shuffled": True,
    }


def facts_sensors(path: Path) -> dict:
    raw = pd.read_csv(path, sep="\t", encoding="utf-8")
    ts = pd.to_datetime(raw["timestamp"], format=TS_FMT)
    df = (raw.assign(timestamp=ts)
          .sort_values(["sensor_id", "timestamp"], kind="mergesort").reset_index(drop=True))
    full = pd.date_range(ts.min(), ts.max(), freq="h")
    rows, missing, gap, const_run, const_start, mean_t, corr_s = {}, {}, {}, {}, {}, {}, {}
    daily = {}
    for sid, g in df.groupby("sensor_id"):
        daily[sid] = _series_dict(
            g.groupby(g["timestamp"].dt.strftime("%Y-%m-%d"))["temperature_c"].mean())
        miss_mask = ~full.isin(pd.DatetimeIndex(g["timestamp"]))
        rows[sid] = len(g)
        missing[sid] = int(miss_mask.sum())
        gap[sid] = _longest_true_run(miss_mask)
        run_len, run_i = _longest_equal_run(g["temperature_c"].tolist())
        const_run[sid] = run_len
        const_start[sid] = g["timestamp"].iloc[run_i].strftime(TS_FMT)
        mean_t[sid] = g["temperature_c"].mean()
        corr_s[sid] = g["temperature_c"].corr(g["humidity_pct"])
    imax = df["temperature_c"].idxmax()
    s1 = df[df["sensor_id"] == "S1"]
    by_hour = df.groupby(df["timestamp"].dt.hour)["temperature_c"].mean()
    stuck = max(const_run, key=const_run.get)
    return {
        "n_rows": len(raw),
        "columns": list(raw.columns),
        "n_sensors": int(raw["sensor_id"].nunique()),
        "is_sorted_by_timestamp_in_file": bool(ts.is_monotonic_increasing),
        "timestamp_min": ts.min().strftime(TS_FMT),
        "timestamp_max": ts.max().strftime(TS_FMT),
        "expected_rows_per_sensor": len(full),
        "rows_per_sensor": rows,
        "missing_hours_per_sensor": missing,
        "longest_gap_hours_per_sensor": gap,
        "longest_constant_temp_run_per_sensor": const_run,
        "longest_constant_temp_run_start_per_sensor": const_start,
        "stuck_sensor": stuck,
        "stuck_sensor_run_hours": const_run[stuck],
        "daily_mean_temp_per_sensor": daily,
        "missing_raw": raw.isna().sum().to_dict(),
        "n_missing_humidity": int(raw["humidity_pct"].isna().sum()),
        "mean_temp_per_sensor": mean_t,
        "max_temp": {
            "value": df.loc[imax, "temperature_c"],
            "sensor_id": df.loc[imax, "sensor_id"],
            "timestamp": df.loc[imax, "timestamp"].strftime(TS_FMT),
        },
        "corr_temp_humidity_pearson_all": df["temperature_c"].corr(df["humidity_pct"]),
        "corr_temp_humidity_pearson_per_sensor": corr_s,
        "daily_mean_temp_S1": _series_dict(
            s1.groupby(s1["timestamp"].dt.strftime("%Y-%m-%d"))["temperature_c"].mean()),
        "hottest_hour_of_day_mean_all": int(by_hour.idxmax()),
    }


# ----------------------------------------------------------------------------
# 4) customers.json - Vietnamese text (UTF-8), mixed types, nulls
# ----------------------------------------------------------------------------
VN_CITIES = ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ"]
VN_CITY_P = [0.32, 0.36, 0.12, 0.10, 0.10]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
SEGMENT_P = [0.55, 0.30, 0.15]
SEGMENT_ORDERS_LAMBDA = {"Consumer": 6, "Corporate": 12, "Home Office": 8}
N_CUST = 500
N_CUST_AGE_STR = 30
N_CUST_SPENT_NULL = 25
N_CUST_EMAIL_NULL = 40


def make_customers(rng: np.random.Generator, path: Path) -> dict:
    n = N_CUST
    names = _vn_names(rng, n)
    city = rng.choice(VN_CITIES, size=n, p=VN_CITY_P)
    seg = rng.choice(SEGMENTS, size=n, p=SEGMENT_P)
    start = pd.Timestamp("2022-01-01")
    span = (pd.Timestamp("2024-12-31") - start).days + 1
    signup = start + pd.to_timedelta(rng.integers(0, span, size=n), unit="D")
    age = rng.integers(18, 71, size=n)
    n_orders = rng.poisson([SEGMENT_ORDERS_LAMBDA[s] for s in seg])
    spent = np.round(n_orders * rng.lognormal(np.log(45), 0.35, size=n), 2)
    active = rng.random(n) < 0.7
    age_str = set(rng.choice(n, N_CUST_AGE_STR, replace=False).tolist())
    spent_null = set(rng.choice(n, N_CUST_SPENT_NULL, replace=False).tolist())
    email_null = set(rng.choice(n, N_CUST_EMAIL_NULL, replace=False).tolist())

    records = []
    for i in range(n):
        records.append({
            "customer_id": f"C{i + 1:04d}",
            "full_name": names[i],
            "city": str(city[i]),
            "signup_date": signup[i].strftime("%Y-%m-%d"),
            "age": str(int(age[i])) if i in age_str else int(age[i]),
            "segment": str(seg[i]),
            "n_orders": int(n_orders[i]),
            "total_spent": None if i in spent_null else float(spent[i]),
            "is_active": bool(active[i]),
            "email": None if i in email_null else f"kh{i + 1:04d}@example.com",
        })
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    return {
        "n_rows": n,
        "n_age_stored_as_string": N_CUST_AGE_STR,
        "n_missing_total_spent": N_CUST_SPENT_NULL,
        "n_missing_email": N_CUST_EMAIL_NULL,
        "encoding": "utf-8",
    }


def facts_customers(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(raw)
    age = pd.to_numeric(df["age"], errors="coerce")
    seg = df.groupby("segment")["total_spent"]
    city = df["city"].value_counts()
    try:
        naive_age_dtype = str(pd.read_json(path, encoding="utf-8")["age"].dtype)
    except Exception as e:  # informational only
        naive_age_dtype = f"error:{type(e).__name__}"
    return {
        "n_rows": len(df),
        "columns": list(df.columns),
        "n_age_stored_as_string": sum(isinstance(r["age"], str) for r in raw),
        "pandas_read_json_age_dtype": naive_age_dtype,
        "mean_age": age.mean(),
        "median_age": age.median(),
        "missing": df.isna().sum().to_dict(),
        "columns_with_missing": [c for c in df.columns if df[c].isna().any()],
        "segment_counts": _vc(df["segment"]),
        "mean_total_spent_by_segment_skipna": _series_dict(seg.mean()),
        "active_share_pct": df["is_active"].mean() * 100,
        "mean_total_spent_by_is_active_skipna": _series_dict(
            df.groupby("is_active")["total_spent"].mean()),
        "customers_by_city": _vc(df["city"]),
        "top_city_by_customers": str(city.idxmax()),
        "corr_n_orders_total_spent_pearson": df["n_orders"].corr(df["total_spent"]),
        "signups_by_year": _vc(df["signup_date"].str[:4]),
    }


# ----------------------------------------------------------------------------
# 5) eu_sales.csv - separator ';', decimal ',', day-first dates dd.mm.yyyy
# ----------------------------------------------------------------------------
EU_COUNTRIES = ["DE", "FR", "IT", "ES", "NL"]
EU_COUNTRY_P = [0.30, 0.22, 0.18, 0.15, 0.15]
EU_PRODUCTS = {"P-A": 19.9, "P-B": 49.5, "P-C": 5.25, "P-D": 120.0, "P-E": 75.8}
EU_PRODUCT_P = [0.25, 0.20, 0.30, 0.10, 0.15]
EU_MONTH_W = [0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 0.85, 0.8, 1.0, 1.1, 1.6, 1.3]
N_EU = 600
EU_DATE_FMT = "%d.%m.%Y"


def make_eu_sales(rng: np.random.Generator, path: Path) -> dict:
    n = N_EU
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    w = np.array([EU_MONTH_W[d.month - 1] for d in days])
    w = w / w.sum()
    dates = pd.DatetimeIndex(np.sort(rng.choice(days.values, size=n, p=w)))
    prod = rng.choice(list(EU_PRODUCTS), size=n, p=EU_PRODUCT_P)
    price = np.array([EU_PRODUCTS[p] for p in prod])
    units = rng.poisson(18, size=n) + 1
    revenue = np.round(units * price * (1 + rng.normal(0, 0.03, size=n)), 2)
    cost = np.round(revenue * rng.uniform(0.55, 0.75, size=n), 2)
    df = pd.DataFrame({
        "date": dates.strftime(EU_DATE_FMT),
        "country": rng.choice(EU_COUNTRIES, size=n, p=EU_COUNTRY_P),
        "product": prod,
        "units": units,
        "revenue_eur": revenue,
        "cost_eur": cost,
    })
    _write_csv(df, path, sep=";", decimal=",")
    return {"n_rows": n, "n_cols": 6, "sep": ";", "decimal": ",", "date_format": EU_DATE_FMT}


def facts_eu_sales(path: Path) -> dict:
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")
    d = pd.to_datetime(df["date"], format=EU_DATE_FMT)
    rev, cost = df["revenue_eur"].sum(), df["cost_eur"].sum()
    by_c = df.groupby("country")["revenue_eur"].sum()
    monthly = df.groupby(d.dt.strftime("%Y-%m"))["revenue_eur"].sum()
    try:
        nv = pd.read_csv(path)
        naive = {"ok": True, "n_cols": int(nv.shape[1]),
                 "n_numeric_cols": int(nv.select_dtypes("number").shape[1])}
    except Exception as e:
        naive = {"ok": False, "error": type(e).__name__}
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "numeric_columns": list(df.select_dtypes("number").columns),
        "naive_read_csv_default": naive,
        "total_revenue_eur": rev,
        "total_cost_eur": cost,
        "gross_margin_pct": (rev - cost) / rev * 100,
        "revenue_by_country": _series_dict(by_c),
        "top_country_by_revenue": str(by_c.idxmax()),
        "units_by_product": _series_dict(df.groupby("product")["units"].sum()),
        "monthly_revenue_eur": _series_dict(monthly),
        "best_month_by_revenue": str(monthly.idxmax()),
        "n_dates_with_day_le_12": int((d.dt.day <= 12).sum()),
    }


# ----------------------------------------------------------------------------
# 6) clinic_messy.xlsx - mixed date formats, impossible dates, invalid values
# ----------------------------------------------------------------------------
N_CLINIC = 400
CLINIC_DATE_KINDS = {"iso_text": 40, "dmy_text": 40, "long_text": 16,
                     "impossible": 9, "garbage": 7, "blank": 12}
CLINIC_IMPOSSIBLE = ["2024-02-30", "31/04/2024", "2024-13-01", "30/02/2024", "2024-04-31",
                     "32/01/2024", "2024-00-10", "29/02/2023", "2024-06-31"]
# Garbage tokens must NOT be in pandas' default na_values ("N/A", "NA", "null",
# "None", "nan", "#N/A", ...), otherwise read_excel silently turns an "invalid"
# value into "missing" and the ground truth depends on reader options.
CLINIC_GARBAGE = ["unknown", "no date", "?", "pending", "--", "not recorded", "TBD"]
AGE_NEG_VALUES = [-5, -1, -30, -2]
AGE_HIGH_VALUES = [999, 150, 200, 999, 135]
N_AGE_UNKNOWN, N_AGE_NUMTEXT, N_AGE_BLANK = 6, 8, 3
N_SEX_BLANK = 10
N_BP_ZERO, BP_HIGH_VALUES, N_BP_BLANK = 4, [400, 500, 999], 5
N_COST_TEXT = 10
AGE_VALID_RANGE = (0, 120)
BP_VALID_RANGE = (50, 250)
CLINIC_DIAG = {"Checkup": 200_000, "Flu": 350_000, "Asthma": 900_000,
               "Hypertension": 1_200_000, "Diabetes": 1_500_000}
CLINIC_DIAG_P = [0.30, 0.25, 0.10, 0.20, 0.15]
_LONG_DATE_RE = re.compile(r"^([A-Za-z]+) (\d{1,2}), (\d{4})$")


def parse_clinic_date(v) -> tuple[str, pd.Timestamp | None]:
    """Canonical rule (also written in sheet 'data_dictionary').

    Returns (status, timestamp) with status in {"valid", "missing", "invalid"}.
    Accepted: Excel date cells, "YYYY-MM-DD", "DD/MM/YYYY", "Month D, YYYY".
    Impossible calendar dates (e.g. 2024-02-30) and free text are invalid.
    """
    if _is_blank(v):
        return "missing", None
    if isinstance(v, datetime):  # pd.Timestamp is a datetime subclass
        return "valid", pd.Timestamp(v)
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return "valid", pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            pass
    m = _LONG_DATE_RE.match(s)
    if m and m.group(1) in MONTHS_EN:
        try:
            return "valid", pd.Timestamp(datetime(int(m.group(3)),
                                                  MONTHS_EN.index(m.group(1)) + 1,
                                                  int(m.group(2))))
        except ValueError:
            return "invalid", None
    return "invalid", None


def normalize_sex(v) -> str | None:
    if _is_blank(v):
        return None
    s = str(v).strip().lower()
    if s in ("m", "male"):
        return "M"
    if s in ("f", "female"):
        return "F"
    return None


def parse_cost_vnd(v) -> int | None:
    """Numbers as-is; text like '1.250.000' uses '.' as thousands separator."""
    if _is_blank(v):
        return None
    if isinstance(v, str):
        s = v.strip().replace(".", "").replace(" ", "")
        return int(s) if s.isdigit() else None
    return int(v)


def make_clinic(rng: np.random.Generator, path: Path) -> dict:
    n = N_CLINIC
    true_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 366, size=n), unit="D")
    visit: list = [true_dates[i].to_pydatetime() for i in range(n)]
    kinds = _disjoint(rng, n, CLINIC_DATE_KINDS)
    for i in kinds["iso_text"]:
        visit[i] = true_dates[i].strftime("%Y-%m-%d")
    for i in kinds["dmy_text"]:
        visit[i] = true_dates[i].strftime("%d/%m/%Y")
    for i in kinds["long_text"]:
        d = true_dates[i]
        visit[i] = f"{MONTHS_EN[d.month - 1]} {d.day}, {d.year}"
    for j, i in enumerate(kinds["impossible"]):
        visit[i] = CLINIC_IMPOSSIBLE[j]
    for j, i in enumerate(kinds["garbage"]):
        visit[i] = CLINIC_GARBAGE[j]
    for i in kinds["blank"]:
        visit[i] = None
    bad_date = np.concatenate([kinds["impossible"], kinds["garbage"], kinds["blank"]])
    valid_date_mask = np.ones(n, dtype=bool)
    valid_date_mask[bad_date] = False

    age_true = rng.integers(1, 96, size=n)
    age: list = [int(a) for a in age_true]
    ak = _disjoint(rng, n, {"neg": len(AGE_NEG_VALUES), "high": len(AGE_HIGH_VALUES),
                            "unknown": N_AGE_UNKNOWN, "numtext": N_AGE_NUMTEXT,
                            "blank": N_AGE_BLANK})
    for j, i in enumerate(ak["neg"]):
        age[i] = AGE_NEG_VALUES[j]
    for j, i in enumerate(ak["high"]):
        age[i] = AGE_HIGH_VALUES[j]
    for i in ak["unknown"]:
        age[i] = "unknown"
    for i in ak["numtext"]:
        age[i] = str(int(age_true[i]))
    for i in ak["blank"]:
        age[i] = None
    age_valid_mask = np.ones(n, dtype=bool)
    age_valid_mask[np.concatenate([ak["neg"], ak["high"], ak["unknown"], ak["blank"]])] = False

    sex_true = rng.choice(["M", "F"], size=n, p=[0.47, 0.53])
    variants = {"M": ["M", "m", "Male", "male"], "F": ["F", "f", "Female", "female"]}
    sex: list = [str(variants[s][rng.integers(0, 4)]) for s in sex_true]
    for i in rng.choice(n, N_SEX_BLANK, replace=False):
        sex[i] = None

    bp_true = np.clip(np.round(rng.normal(125, 15, size=n)), 90, 190).astype(int)
    bp: list = [int(b) for b in bp_true]
    bk = _disjoint(rng, n, {"zero": N_BP_ZERO, "high": len(BP_HIGH_VALUES), "blank": N_BP_BLANK})
    for i in bk["zero"]:
        bp[i] = 0
    for j, i in enumerate(bk["high"]):
        bp[i] = BP_HIGH_VALUES[j]
    for i in bk["blank"]:
        bp[i] = None

    diag = rng.choice(list(CLINIC_DIAG), size=n, p=CLINIC_DIAG_P)
    cost_true = np.round(np.array([CLINIC_DIAG[d] for d in diag])
                         * rng.lognormal(0, 0.3, size=n), -3).astype(int)
    cost: list = [int(c) for c in cost_true]
    for i in rng.choice(n, N_COST_TEXT, replace=False):
        cost[i] = f"{int(cost_true[i]):,}".replace(",", ".")

    df = pd.DataFrame({
        "patient_id": [f"P{i + 1:04d}" for i in range(n)],
        "visit_date": pd.Series(visit, dtype=object),
        "age": pd.Series(age, dtype=object),
        "sex": pd.Series(sex, dtype=object),
        "systolic_bp": pd.Series(bp, dtype=object),
        "diagnosis": diag,
        "cost_vnd": pd.Series(cost, dtype=object),
    })
    dictionary = pd.DataFrame({
        "column": ["patient_id", "visit_date", "age", "sex", "systolic_bp", "diagnosis", "cost_vnd"],
        "description": [
            "Patient identifier.",
            "Visit date, entered manually. Text dates use DD/MM/YYYY (day first). "
            "Impossible dates and free text are invalid.",
            f"Age in years. Valid range {AGE_VALID_RANGE[0]}-{AGE_VALID_RANGE[1]}; "
            "numbers stored as text are valid.",
            "Sex, mixed spellings (M/m/Male/male, F/f/Female/female).",
            f"Systolic blood pressure in mmHg. Valid range {BP_VALID_RANGE[0]}-{BP_VALID_RANGE[1]}.",
            "Diagnosis category.",
            "Cost in VND. Text values use '.' as the thousands separator.",
        ],
    })
    with pd.ExcelWriter(path, engine="openpyxl", date_format="YYYY-MM-DD",
                        datetime_format="YYYY-MM-DD") as xw:
        df.to_excel(xw, sheet_name="patients", index=False)
        dictionary.to_excel(xw, sheet_name="data_dictionary", index=False)

    valid_months = pd.Series(true_dates[valid_date_mask].strftime("%Y-%m")).value_counts()
    return {
        "n_rows": n,
        "visit_date_kinds": {**CLINIC_DATE_KINDS, "excel_date": n - sum(CLINIC_DATE_KINDS.values())},
        "n_visit_date_missing": CLINIC_DATE_KINDS["blank"],
        "n_visit_date_invalid": CLINIC_DATE_KINDS["impossible"] + CLINIC_DATE_KINDS["garbage"],
        "n_visit_date_valid": int(valid_date_mask.sum()),
        "n_visit_date_valid_text": (CLINIC_DATE_KINDS["iso_text"] + CLINIC_DATE_KINDS["dmy_text"]
                                    + CLINIC_DATE_KINDS["long_text"]),
        "visits_per_month_valid": _series_dict(valid_months),
        "n_age_missing": N_AGE_BLANK,
        "n_age_invalid": len(AGE_NEG_VALUES) + len(AGE_HIGH_VALUES) + N_AGE_UNKNOWN,
        "n_age_numeric_text": N_AGE_NUMTEXT,
        "mean_age_valid": float(age_true[age_valid_mask].mean()),
        "n_sex_missing": N_SEX_BLANK,
        "n_bp_missing": N_BP_BLANK,
        "n_bp_invalid": N_BP_ZERO + len(BP_HIGH_VALUES),
        "n_cost_text": N_COST_TEXT,
        "total_cost_vnd": int(cost_true.sum()),
    }


def facts_clinic(path: Path) -> dict:
    sheets = pd.ExcelFile(path, engine="openpyxl").sheet_names
    df = pd.read_excel(path, sheet_name="patients", engine="openpyxl")
    parsed = [parse_clinic_date(v) for v in df["visit_date"]]
    status = pd.Series([p[0] for p in parsed])
    dates = pd.Series([p[1] for p in parsed], dtype="datetime64[ns]")
    is_text = df["visit_date"].map(lambda v: isinstance(v, str))
    valid_dates = dates[status == "valid"]
    months = valid_dates.dt.strftime("%Y-%m").value_counts()

    age_blank = df["age"].map(_is_blank)
    age_num = pd.to_numeric(df["age"], errors="coerce")
    age_ok = ~age_blank & age_num.between(*AGE_VALID_RANGE)
    age_numtext = df["age"].map(lambda v: isinstance(v, str) and v.strip().lstrip("-").isdigit())

    sex = df["sex"].map(normalize_sex)
    bp_blank = df["systolic_bp"].map(_is_blank)
    bp_num = pd.to_numeric(df["systolic_bp"], errors="coerce")
    bp_ok = ~bp_blank & bp_num.between(*BP_VALID_RANGE)
    cost = df["cost_vnd"].map(parse_cost_vnd)
    clean = df.assign(cost_num=cost.astype("float64"), bp_num=bp_num.where(bp_ok))
    cost_by_diag = clean.groupby("diagnosis")["cost_num"].mean()
    return {
        "sheet_names": sheets,
        "n_rows": len(df),
        "columns": list(df.columns),
        "visit_date_dtype_raw": str(df["visit_date"].dtype),
        "n_visit_date_missing": int((status == "missing").sum()),
        "n_visit_date_invalid": int((status == "invalid").sum()),
        "n_visit_date_valid": int((status == "valid").sum()),
        "n_visit_date_valid_text": int(((status == "valid") & is_text).sum()),
        "invalid_visit_date_values": sorted(str(v) for v, s in zip(df["visit_date"], status)
                                            if s == "invalid"),
        "visits_per_month_valid": _series_dict(months),
        "busiest_month_valid": str(months.sort_index().idxmax()),
        "n_age_missing": int(age_blank.sum()),
        "n_age_invalid": int((~age_blank & ~age_ok).sum()),
        "n_age_valid": int(age_ok.sum()),
        "n_age_numeric_text": int(age_numtext.sum()),
        "mean_age_valid": age_num[age_ok].mean(),
        "sex_raw_distinct_values": sorted(str(v) for v in df["sex"].dropna().unique()),
        "sex_counts_normalized": {"F": int((sex == "F").sum()), "M": int((sex == "M").sum()),
                                  "unknown": int(sex.isna().sum())},
        "n_sex_missing": int(df["sex"].map(_is_blank).sum()),
        "n_bp_missing": int(bp_blank.sum()),
        "n_bp_invalid": int((~bp_blank & ~bp_ok).sum()),
        "mean_bp_valid": bp_num[bp_ok].mean(),
        "mean_bp_valid_by_diagnosis": _series_dict(clean.groupby("diagnosis")["bp_num"].mean()),
        "n_cost_text": int(df["cost_vnd"].map(lambda v: isinstance(v, str)).sum()),
        "total_cost_vnd": int(cost.sum()),
        "mean_cost_vnd_by_diagnosis": _series_dict(cost_by_diag),
        "top_diagnosis_by_mean_cost": str(cost_by_diag.idxmax()),
        "diagnosis_counts": _vc(df["diagnosis"]),
    }


# ----------------------------------------------------------------------------
# 7) transactions.parquet - categorical dtypes, confounded correlation
# ----------------------------------------------------------------------------
# name: (daily count at 24C, temperature slope, quantity range, unit price)
TX_CATEGORIES = {
    "ice_cream": (12, 0.09, (1, 4), 1.8),
    "fan": (8, 0.10, (1, 2), 25.0),
    "sunscreen": (6, 0.08, (1, 2), 9.5),
    "bread": (30, 0.0, (1, 3), 1.2),
    "milk": (28, 0.0, (1, 3), 1.5),
    "coffee": (35, 0.0, (1, 2), 2.5),
}
TX_TEMP_DRIVEN = ["ice_cream", "fan", "sunscreen"]
TX_STORES = [f"ST{i:02d}" for i in range(1, 21)]
TX_PAY = ["cash", "card", "e-wallet"]
TX_PAY_P = [0.30, 0.45, 0.25]


def make_transactions(rng: np.random.Generator, path: Path) -> dict:
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    doy = days.dayofyear.values
    temp = np.round(24 + 7 * np.sin(2 * np.pi * (doy - 110) / 366)
                    + rng.normal(0, 2.0, size=len(days)), 1)
    day_parts, cat_parts = [], []
    for cat, (base, slope, _, _) in TX_CATEGORIES.items():
        cnt = rng.poisson(base * np.exp(slope * (temp - 24)))
        day_parts.append(np.repeat(np.arange(len(days)), cnt))
        cat_parts.append(np.repeat(cat, cnt.sum()))
    day_idx = np.concatenate(day_parts)
    cat = np.concatenate(cat_parts)
    n = len(day_idx)

    secs = rng.integers(7 * 3600, 22 * 3600, size=n)
    ts = days.values[day_idx] + secs.astype("timedelta64[s]")
    qty = np.empty(n, dtype=np.int32)
    price = np.empty(n, dtype=float)
    for c, (_, _, (lo, hi), p) in TX_CATEGORIES.items():
        m = cat == c
        qty[m] = rng.integers(lo, hi + 1, size=m.sum())
        price[m] = np.round(p * (1 + rng.normal(0, 0.03, size=m.sum())), 2)
    store_p = rng.uniform(0.5, 1.5, size=len(TX_STORES))
    store_p = store_p / store_p.sum()

    df = pd.DataFrame({
        "ts": ts,
        "store_id": rng.choice(TX_STORES, size=n, p=store_p),
        "product_category": cat,
        "quantity": qty,
        "unit_price": price,
        "amount": np.round(qty * price, 2),
        "payment_method": rng.choice(TX_PAY, size=n, p=TX_PAY_P),
        "temp_c": temp[day_idx],
    })
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    df.insert(0, "txn_id", np.arange(1, n + 1, dtype=np.int64))
    for c in ("store_id", "product_category", "payment_method"):
        df[c] = df[c].astype("category")
    df.to_parquet(path, engine="pyarrow", index=False)
    return {
        "n_rows": n,
        "n_days": len(days),
        "n_stores": len(TX_STORES),
        "confounder": "temp_c",
        "temperature_driven_categories": TX_TEMP_DRIVEN,
        "flat_categories": [c for c in TX_CATEGORIES if c not in TX_TEMP_DRIVEN],
    }


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    coef = np.polyfit(x, y, 1)
    return y - np.polyval(coef, x)


def facts_transactions(path: Path) -> dict:
    df = pd.read_parquet(path, engine="pyarrow")
    day = df["ts"].dt.normalize()
    full_days = pd.date_range(day.min(), day.max(), freq="D")
    daily = (df.assign(day=day)
             .pivot_table(index="day", columns="product_category", values="quantity",
                          aggfunc="sum", fill_value=0, observed=True)
             .reindex(full_days, fill_value=0))
    daily.columns = [str(c) for c in daily.columns]
    temp = df.assign(day=day).groupby("day")["temp_c"].first().reindex(full_days)
    ok = temp.notna().to_numpy()
    t = temp.to_numpy()[ok]
    ice = daily["ice_cream"].to_numpy(dtype=float)[ok]
    fan = daily["fan"].to_numpy(dtype=float)[ok]
    bread = daily["bread"].to_numpy(dtype=float)[ok]

    rev_cat = df.groupby("product_category", observed=True)["amount"].sum()
    rev_store = df.groupby("store_id", observed=True)["amount"].sum()
    return {
        "n_rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(tp) for c, tp in df.dtypes.items()},
        "n_stores": int(df["store_id"].nunique()),
        "n_categories": int(df["product_category"].nunique()),
        "n_days": len(full_days),
        "mean_daily_temp": float(t.mean()),
        "ts_min": df["ts"].min().isoformat(),
        "ts_max": df["ts"].max().isoformat(),
        "total_amount": df["amount"].sum(),
        "revenue_by_category": _series_dict(rev_cat),
        "units_by_category": _series_dict(df.groupby("product_category", observed=True)["quantity"].sum()),
        "mean_daily_units_by_category": {c: daily[c].mean() for c in daily.columns},
        "revenue_by_store": _series_dict(rev_store),
        "top_store_by_revenue": str(rev_store.idxmax()),
        "payment_share_pct": _series_dict(df["payment_method"].value_counts(normalize=True) * 100),
        "busiest_hour_by_txn_count": int(df.groupby(df["ts"].dt.hour).size().idxmax()),
        "monthly_revenue": _series_dict(df.groupby(df["ts"].dt.strftime("%Y-%m"))["amount"].sum()),
        "corr_daily_units_icecream_fan": float(np.corrcoef(ice, fan)[0, 1]),
        "corr_daily_units_icecream_temp": float(np.corrcoef(ice, t)[0, 1]),
        "corr_daily_units_fan_temp": float(np.corrcoef(fan, t)[0, 1]),
        "corr_daily_units_bread_icecream": float(np.corrcoef(bread, ice)[0, 1]),
        "partial_corr_icecream_fan_given_temp": float(
            np.corrcoef(_residual(ice, t), _residual(fan, t))[0, 1]),
    }


# ----------------------------------------------------------------------------
# Registry + build
# ----------------------------------------------------------------------------
DATASETS = [
    {"name": "sales", "seed_idx": 1, "file": "sales.csv", "format": "csv",
     "make": make_sales, "facts": facts_sales,
     "read_hint": "pd.read_csv(path)",
     "traps": ["25 exact duplicate rows, adjacent to their originals",
               "missing region (40) and discount (60) in unique rows",
               "8 genuine bulk stationery orders (quantity 300-600): quantity outliers",
               "revenue is heavily right-skewed (Laptop/Phone); IQR flags many orders",
               "Q4 seasonality; 2024 has +5% prices and about +10% volume"]},
    {"name": "hr", "seed_idx": 2, "file": "hr.xlsx", "format": "xlsx",
     "make": make_hr, "facts": facts_hr,
     "read_hint": 'pd.read_excel(path, sheet_name="employees") and sheet_name="departments"',
     "traps": ["first sheet is README (notes), data is on sheets 'employees' and 'departments'",
               "5 employees have dept_id 99 which does not exist in 'departments'",
               "tenure must use the fixed reference date 2025-01-01, not today"]},
    {"name": "sensors", "seed_idx": 3, "file": "sensors.tsv", "format": "tsv",
     "make": make_sensors, "facts": facts_sensors,
     "read_hint": 'pd.read_csv(path, sep="\\t")',
     "traps": ["rows are shuffled: must sort by timestamp before time plots",
               "S2 has a 12-hour outage; S3 has 20 random missing hours",
               "S1 temperature is frozen for 10 consecutive hours",
               "30 missing humidity values"]},
    {"name": "customers", "seed_idx": 4, "file": "customers.json", "format": "json",
     "make": make_customers, "facts": facts_customers,
     "read_hint": "pd.read_json(path) or pd.DataFrame(json.load(open(path, encoding='utf-8')))",
     "traps": ["UTF-8 Vietnamese text (names, cities)",
               "30 ages stored as strings",
               "null total_spent (25) and email (40)"]},
    {"name": "eu_sales", "seed_idx": 5, "file": "eu_sales.csv", "format": "csv",
     "make": make_eu_sales, "facts": facts_eu_sales,
     "read_hint": 'pd.read_csv(path, sep=";", decimal=",")',
     "traps": ["separator ';' and decimal ','; default read_csv breaks",
               "dates are day-first dd.mm.yyyy"]},
    {"name": "clinic", "seed_idx": 6, "file": "clinic_messy.xlsx", "format": "xlsx",
     "make": make_clinic, "facts": facts_clinic,
     "read_hint": 'pd.read_excel(path, sheet_name="patients")',
     "traps": ["visit_date mixes Excel dates, ISO text, DD/MM/YYYY text and 'Month D, YYYY'",
               "9 impossible dates, 7 free-text dates, 12 blank dates",
               "age: negatives, >120, 'unknown', numbers stored as text, blanks",
               "sex has mixed spellings; systolic_bp has 0 and >250 values",
               "cost_vnd text values use '.' as thousands separator",
               "rules are documented in sheet 'data_dictionary'"]},
    {"name": "transactions", "seed_idx": 7, "file": "transactions.parquet", "format": "parquet",
     "make": make_transactions, "facts": facts_transactions,
     "read_hint": "pd.read_parquet(path)",
     "traps": ["daily ice_cream and fan sales are strongly correlated only because both follow "
               "temperature (temp_c); correlation is not causation",
               "categorical dtypes: use observed=True in groupby"]},
]


def build_all(out_dir: str | Path, seed: int = DEFAULT_SEED) -> dict:
    out_dir = Path(out_dir).resolve()
    syn = out_dir / SYNTH_DIR
    syn.mkdir(parents=True, exist_ok=True)
    gt: dict = {
        "meta": {
            "gt_version": GT_VERSION,
            "seed": seed,
            "reference_date": REFERENCE_DATE,
            "generator": "datasets/make_datasets.py",
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "float_digits": FLOAT_DIGITS,
        },
        "datasets": {},
    }
    for spec in DATASETS:
        rng = np.random.default_rng([seed, spec["seed_idx"]])
        path = syn / spec["file"]
        injected = spec["make"](rng, path)
        facts = spec["facts"](path)
        gt["datasets"][spec["name"]] = {
            "file": f"{SYNTH_DIR}/{spec['file']}",
            "format": spec["format"],
            "read_hint": spec["read_hint"],
            "traps": spec["traps"],
            "injected": injected,
            "facts": facts,
        }
    text = json.dumps(_clean(gt), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (out_dir / GT_FILE).write_text(text, encoding="utf-8", newline="\n")
    return json.loads(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic datasets + ground truth.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent),
                    help="output folder (default: the datasets/ folder)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    gt = build_all(args.out, args.seed)
    out = Path(args.out).resolve()
    print(f"{'dataset':<13} {'format':<8} {'size':>10}  file")
    for name, d in gt["datasets"].items():
        size_kb = (out / d["file"]).stat().st_size / 1024
        print(f"{name:<13} {d['format']:<8} {size_kb:8.1f} KB  {d['file']}")
    print(f"ground truth: {out / GT_FILE} (seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
