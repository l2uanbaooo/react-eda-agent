"""Shared pytest setup: make project modules importable without packaging.

datasets/ is deliberately NOT a Python package: on Kaggle the HuggingFace
`datasets` library is pre-installed and `import datasets` would be ambiguous.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "datasets", ROOT / "eval", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
