"""Download the real demo datasets ONCE (the Docker sandbox has no network).

Usage (from the project root, Windows):
    .venv\\Scripts\\python.exe datasets\\fetch_real.py           # download missing files
    .venv\\Scripts\\python.exe datasets\\fetch_real.py --verify  # only check checksums
    .venv\\Scripts\\python.exe datasets\\fetch_real.py --force   # re-download everything

Writes datasets/real/<name>.csv and datasets/real/SOURCES.json with url, sha256,
size, rows, cols and fetch date. On later runs the sha256 recorded at first
download is verified, so a silently changed upstream file is detected.
These files are for the demo and a few aggregate tasks; they have no planted
traps and no generated ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

REAL_DIR = Path(__file__).resolve().parent / "real"
SOURCES_FILE = REAL_DIR / "SOURCES.json"
USER_AGENT = "eda-agent-dataset-fetch/1.0"

SEABORN = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master"
REAL_DATASETS = {
    "titanic.csv": {
        "url": f"{SEABORN}/titanic.csv",
        "origin": "seaborn-data repository (mwaskom/seaborn-data)",
        "description": "Titanic passengers: survival, class, sex, age, fare.",
    },
    "penguins.csv": {
        "url": f"{SEABORN}/penguins.csv",
        "origin": "seaborn-data repository; Palmer Penguins (Horst, Hill, Gorman), CC0",
        "description": "Penguin species, island, bill/flipper size, body mass.",
    },
    "tips.csv": {
        "url": f"{SEABORN}/tips.csv",
        "origin": "seaborn-data repository",
        "description": "Restaurant bills and tips by day, time, party size.",
    },
    "california_housing.csv": {
        "url": "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
        "origin": "ageron/handson-ml2 repository; 1990 California census (Pace & Barry)",
        "description": "Block-group housing values, income, rooms, ocean proximity.",
    },
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            wait = 2 ** attempt
            print(f"  download failed ({e}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"could not download {url}: {last}")


def _load_sources() -> dict:
    if SOURCES_FILE.exists():
        return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    return {}


def _save_sources(sources: dict) -> None:
    text = json.dumps(sources, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    SOURCES_FILE.write_text(text, encoding="utf-8", newline="\n")


def verify() -> bool:
    sources = _load_sources()
    ok = True
    for name in REAL_DATASETS:
        path = REAL_DIR / name
        rec = sources.get(name)
        if not path.exists() or rec is None:
            print(f"MISSING  {name}")
            ok = False
            continue
        actual = _sha256(path.read_bytes())
        status = "OK" if actual == rec["sha256"] else "CHANGED"
        ok = ok and status == "OK"
        print(f"{status:<8} {name}  {rec['bytes']} bytes  {rec['rows']}x{rec['cols']}")
    return ok


def fetch(force: bool = False) -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    sources = _load_sources()
    for name, meta in REAL_DATASETS.items():
        path = REAL_DIR / name
        if path.exists() and name in sources and not force:
            print(f"skip     {name} (already downloaded; use --verify or --force)")
            continue
        print(f"download {name} <- {meta['url']}")
        data = _download(meta["url"])
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        df = pd.read_csv(tmp, encoding="utf-8")  # fail before replacing a good file
        tmp.replace(path)
        sources[name] = {
            **meta,
            "sha256": _sha256(data),
            "bytes": len(data),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "fetched_on": date.today().isoformat(),
        }
        print(f"         {len(data)} bytes, {df.shape[0]} rows x {df.shape[1]} cols")
    _save_sources(sources)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download real demo datasets once.")
    ap.add_argument("--force", action="store_true", help="re-download all files")
    ap.add_argument("--verify", action="store_true", help="only verify checksums")
    args = ap.parse_args(argv)
    if args.verify:
        return 0 if verify() else 1
    fetch(force=args.force)
    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
