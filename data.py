"""The ONE data loader + automatic pre-profiling.

Used everywhere: the agent host (profiling for the prompt), the sandbox (this
file is bind-mounted read-only at /opt/eda/data.py and imported by the
worker), the Streamlit UI and the report. Only pandas/numpy/openpyxl/stdlib.

    load_data(path, sheet=None, *, sep=None, decimal=None, encoding=None) -> DataFrame
    list_sheets(path) -> [{"name", "rows", "cols"}]       (xlsx only, else [])
    profile(df, max_chars=6000) -> str                      (English, for the LLM prompt)
    profile_file(path, sheet=None, max_chars=6000) -> str   (adds the sheet list)

Formats: csv / txt (delimiter and decimal comma sniffed), tsv, xlsx/xlsm (sheet
by name or index), json (records, columns dict, or {"key": [records]}),
jsonl/ndjson, parquet.

Excel sheet when sheet=None (policy, one function everywhere):
  "largest" (default: UI, report) - the sheet with the most cells;
  "first"   (agent sandbox, all eval configs; set via EDA_DEFAULT_SHEET by the
             sandbox) - the first sheet, exactly like pd.read_excel, so the
             loader itself never hints which sheet holds the data.
df.attrs["source"] records the loaded sheet only, never the list of sheets.
JSON is loaded with json.load + DataFrame so raw types are kept: a column
holding 34 and "34" stays object (pd.read_json would silently coerce it).
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

FORMATS = {
    ".csv": "csv", ".txt": "csv", ".tsv": "tsv",
    ".xlsx": "excel", ".xlsm": "excel",
    ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl",
    ".parquet": "parquet", ".pq": "parquet",
}
_CANDIDATE_SEPS = [",", ";", "\t", "|"]
_DEC_COMMA_RE = re.compile(r"^-?\d+,\d+$")
_DEC_DOT_RE = re.compile(r"^-?\d+\.\d+$")
_DATE_CHARS_RE = re.compile(r"[-/.:]|[A-Za-z]{3,}")


class DataLoadError(ValueError):
    """Raised with a message meant to be read by a human or by the agent."""


# ----------------------------------------------------------------------------
# Format / encoding / delimiter detection
# ----------------------------------------------------------------------------
def detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".xls":
        raise DataLoadError("Legacy .xls is not supported; save the file as .xlsx")
    if suffix not in FORMATS:
        raise DataLoadError(f"Unsupported file type {suffix!r}; supported: {sorted(FORMATS)}")
    return FORMATS[suffix]


def _read_head(path: Path, n: int = 65536) -> tuple[str, str]:
    """Decode the first n bytes; returns (text, encoding for pandas)."""
    raw = path.read_bytes()[:n]
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        for cut in range(4):  # the cut may split a multi-byte UTF-8 character
            try:
                return raw[:len(raw) - cut].decode(enc), enc
            except UnicodeDecodeError:
                continue
    return raw.decode("latin-1"), "latin-1"


def sniff_csv(text: str) -> tuple[str, str]:
    """Return (sep, decimal) from the first lines of a delimited text file."""
    lines = [ln for ln in text.splitlines()[:50] if ln.strip()]
    if not lines:
        return ",", "."
    best, best_score = ",", (-1.0, -1)
    for sep in _CANDIDATE_SEPS:
        counts = [ln.count(sep) for ln in lines]
        if counts[0] == 0:
            continue
        consistency = sum(c == counts[0] for c in counts) / len(counts)
        score = (consistency, counts[0])
        if score > best_score:
            best, best_score = sep, score
    decimal = "."
    if best != ",":
        comma = dot = 0
        for ln in lines[1:]:
            for tok in ln.split(best):
                tok = tok.strip().strip('"')
                comma += bool(_DEC_COMMA_RE.match(tok))
                dot += bool(_DEC_DOT_RE.match(tok))
        if comma > dot:
            decimal = ","
    return best, decimal


# ----------------------------------------------------------------------------
# Excel sheets
# ----------------------------------------------------------------------------
def list_sheets(path: str | Path) -> list[dict]:
    path = Path(path)
    if detect_format(path) != "excel":
        return []
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        out = []
        for ws in wb.worksheets:
            rows, cols = ws.max_row, ws.max_column
            if rows is None or cols is None:  # unknown dimensions: count
                rows = sum(1 for _ in ws.iter_rows())
                cols = max((len(r) for r in ws.iter_rows()), default=0)
            out.append({"name": ws.title, "rows": max(int(rows) - 1, 0), "cols": int(cols)})
        return out
    finally:
        wb.close()


SHEET_POLICIES = ("largest", "first")


def default_sheet_policy() -> str:
    policy = os.environ.get("EDA_DEFAULT_SHEET", "largest").strip().lower()
    if policy not in SHEET_POLICIES:
        raise DataLoadError(f"EDA_DEFAULT_SHEET must be one of {SHEET_POLICIES}, got {policy!r}")
    return policy


def _pick_sheet(path: Path, sheet: str | int | None, policy: str | None = None) -> str:
    sheets = list_sheets(path)
    names = [s["name"] for s in sheets]
    if not names:
        raise DataLoadError(f"{path.name}: workbook has no sheets")
    if sheet is None:
        policy = policy or default_sheet_policy()
        if policy == "first":
            return names[0]
        return max(sheets, key=lambda s: (s["rows"] * s["cols"], -names.index(s["name"])))["name"]
    if isinstance(sheet, int):
        if not 0 <= sheet < len(names):
            raise DataLoadError(f"{path.name}: sheet index {sheet} out of range; sheets: {names}")
        return names[sheet]
    if sheet not in names:
        raise DataLoadError(f"{path.name}: no sheet named {sheet!r}; sheets: {names}")
    return sheet


# ----------------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------------
def _json_frame(path: Path) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        lists = {k: v for k, v in obj.items() if isinstance(v, list)}
        if len(lists) == 1 and all(isinstance(r, dict) for r in next(iter(lists.values()))):
            return pd.DataFrame(next(iter(lists.values())))  # {"data": [records]}
        if lists and len(lists) == len(obj):
            return pd.DataFrame(obj)  # {"col": [values], ...}
    raise DataLoadError(f"{path.name}: unsupported JSON layout (expected a list of records "
                        "or a dict of columns)")


# ----------------------------------------------------------------------------
# Public loader
# ----------------------------------------------------------------------------
def load_data(path: str | Path, sheet: str | int | None = None, *, sep: str | None = None,
              decimal: str | None = None, encoding: str | None = None,
              sheet_policy: str | None = None) -> pd.DataFrame:
    """Load any supported file into a DataFrame. df.attrs["source"] says how.

    sheet_policy ("largest"/"first") only matters for Excel with sheet=None;
    None = environment variable EDA_DEFAULT_SHEET, else "largest".
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")
    fmt = detect_format(path)
    info: dict = {"file": path.name, "format": fmt}
    if fmt in ("csv", "tsv"):
        head, enc = _read_head(path)
        s, d = sniff_csv(head)
        if fmt == "tsv":
            s = "\t"
        sep, decimal, encoding = sep or s, decimal or d, encoding or enc
        df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding, low_memory=False)
        info.update(sep=sep, decimal=decimal, encoding=encoding)
    elif fmt == "excel":
        name = _pick_sheet(path, sheet, sheet_policy)
        df = pd.read_excel(path, sheet_name=name, engine="openpyxl")
        info.update(sheet=name)
    elif fmt == "json":
        df = _json_frame(path)
    elif fmt == "jsonl":
        df = pd.read_json(path, lines=True, dtype=False, encoding="utf-8")
    else:
        df = pd.read_parquet(path)
    df.attrs["source"] = info
    return df


# ----------------------------------------------------------------------------
# Pre-profiling (deterministic text for the prompt)
# ----------------------------------------------------------------------------
def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NaN"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    s = str(v)
    return s if len(s) <= 40 else s[:37] + "..."


def _looks_like_dates(values: pd.Series) -> bool:
    strs = [v for v in values.dropna().head(200) if isinstance(v, str) and len(v) >= 6
            and _DATE_CHARS_RE.search(v) and any(ch.isdigit() for ch in v)]
    if len(strs) < 5:
        return False
    parsed = pd.to_datetime(pd.Series(strs), errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.8


def _column_lines(df: pd.DataFrame) -> list[str]:
    n = len(df)
    lines = ["COLUMNS (name | dtype | missing | unique | examples | notes):"]
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        try:
            nun = int(s.nunique(dropna=True))
        except TypeError:  # unhashable values (lists/dicts from JSON)
            nun = -1
        examples = ", ".join(_fmt(v) for v in s.dropna().unique()[:3]) if nun >= 0 else "-"
        notes = []
        if s.dtype == object:
            types = Counter(type(v).__name__ for v in s.dropna())
            if len(types) > 1:
                notes.append("mixed types: " + ", ".join(f"{t}={k}" for t, k in types.most_common()))
            if _looks_like_dates(s):
                notes.append("text that looks like dates")
        if nun == n and n > 0:
            notes.append("all values unique (id-like)")
        pct = f"{miss / n:.1%}" if n else "0%"
        lines.append(f"- {c} | {s.dtype} | {miss} ({pct}) | {nun} | {examples}"
                     + (f" | {'; '.join(notes)}" if notes else ""))
    return lines


def profile(df: pd.DataFrame, max_chars: int = 6000, top_k: int = 5) -> str:
    out = [f"SHAPE: {len(df)} rows x {df.shape[1]} columns"]
    src = df.attrs.get("source")
    if src:
        out.append("SOURCE: " + ", ".join(f"{k}={v!r}" for k, v in src.items() if k != "sheets"))
    out += _column_lines(df)

    num = df.select_dtypes(include="number").loc[:, lambda x: ~x.columns.duplicated()]
    if not num.empty:
        out.append("NUMERIC SUMMARY (min | p25 | median | mean | p75 | max | std):")
        for c in num.columns:
            s = num[c].dropna()
            if s.empty:
                continue
            q = s.quantile([0.25, 0.5, 0.75])
            out.append(f"- {c}: {_fmt(s.min())} | {_fmt(q[0.25])} | {_fmt(q[0.5])} | "
                       f"{_fmt(s.mean())} | {_fmt(q[0.75])} | {_fmt(s.max())} | {_fmt(s.std())}")

    cat_lines = []
    for c in df.columns:
        s = df[c]
        if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == bool:
            try:
                vc = s.astype(str).where(s.notna()).value_counts()
            except TypeError:
                continue
            if 0 < len(vc) <= 50:
                top = ", ".join(f"{_fmt(k)}={v}" for k, v in vc.head(top_k).items())
                cat_lines.append(f"- {c} ({len(vc)} values): {top}")
    if cat_lines:
        out.append(f"CATEGORICAL TOP {top_k}:")
        out += cat_lines

    if num.shape[1] >= 2:
        corr = num.iloc[:, :30].corr(numeric_only=True)
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iat[i, j]
                if pd.notna(r):
                    pairs.append((abs(r), cols[i], cols[j], r))
        pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
        if pairs:
            out.append("TOP PEARSON CORRELATIONS (correlation is not causation):")
            out += [f"- {a} ~ {b}: {r:.3f}" for _, a, b, r in pairs[:8]]

    text = "\n".join(out)
    if len(text) > max_chars:
        text = text[:max_chars - 30].rsplit("\n", 1)[0] + "\n... [profile truncated]"
    return text


def profile_file(path: str | Path, sheet: str | int | None = None, max_chars: int = 6000) -> str:
    """Profile for the prompt. Excel: lists ALL sheets and profiles the largest
    one (or `sheet`); this sheet list is the only sheet hint the agent gets."""
    path = Path(path)
    head = []
    sheets = list_sheets(path)
    if sheets:
        head.append("SHEETS: " + "; ".join(f"{s['name']} ({s['rows']} rows x {s['cols']} cols)"
                                             for s in sheets))
    df = load_data(path, sheet, sheet_policy="largest")
    if sheets:
        name = df.attrs["source"]["sheet"]
        head.append(f"PROFILED SHEET: {name} (load it with load_data(path, sheet={name!r}))")
    body = profile(df, max_chars=max(500, max_chars - sum(len(h) + 1 for h in head)))
    return "\n".join(head + [body])
