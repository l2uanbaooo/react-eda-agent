"""GD3: the Kaggle notebook is valid and contains the critical settings."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "kaggle" / "ollama_kaggle.ipynb"


def _source() -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["cells"]
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code") and "source" in c
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_notebook_is_valid_and_complete():
    src = _source()
    for needle in ("qwen2.5-coder:14b", "qwen2.5:14b-instruct",  # both models pulled
                   "OLLAMA_KEEP_ALIVE=\"24h\"",                     # model stays loaded
                   "--http-host-header",                          # avoids HTTP 403 through the tunnel
                   "CLOUDFLARED_VERSION = \"2024.12.2\"",         # pinned binary
                   "trycloudflare",                               # URL is extracted and printed
                   "EDA_OLLAMA_URL",                              # tells the user what to set locally
                   "EDA_SANDBOX_MODE",                            # mode B instructions
                   "pip install -q -r requirements.txt"):         # mode B uses the pinned versions
        assert needle in src, needle


def test_notebook_has_no_outputs_committed():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
