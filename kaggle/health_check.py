"""Check that the Ollama server (Kaggle, via tunnel) is ready BEFORE a run.

Usage (local Windows, mode A):
    $env:EDA_OLLAMA_URL = "https://xxxx.trycloudflare.com"
    .venv\\Scripts\\python.exe kaggle\\health_check.py
    .venv\\Scripts\\python.exe kaggle\\health_check.py --judge      # also the judge model
    .venv\\Scripts\\python.exe kaggle\\health_check.py --url http://localhost:11434

Checks: server reachable, version, model pulled, one short warm-up chat
(loads the model into GPU memory), then a ~100-token generation to measure
speed. Exit code 0 = ready, 1 = not ready (message says why).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config  # noqa: E402
from llm import LLMError, OllamaClient  # noqa: E402


def check_model(cfg, model: str, temperature: float, speed: bool) -> bool:
    client = OllamaClient(cfg, model=model, temperature=temperature)
    h = client.health(warmup=True)
    print(f"\n== {model} @ {h['url']}")
    print(f"server version : {h.get('version', '-')}")
    print(f"models on server: {', '.join(h.get('models', [])) or '-'}")
    if not h["ok"]:
        print(f"NOT READY      : {h.get('error')}")
        return False
    print(f"warm-up        : {h['warmup_s']:.1f} s, reply={h['warmup_reply']!r}")
    if speed:
        try:
            res = client.chat([{"role": "user", "content":
                                "Write a Python function that returns the mean of a list. Code only."}],
                              num_predict=100)
        except LLMError as e:
            print(f"NOT READY      : speed test failed: {e}")
            return False
        tps = (res.completion_tokens or 0) / res.duration_s if res.duration_s else 0.0
        print(f"speed          : {res.completion_tokens} tokens in {res.duration_s:.1f} s "
              f"(~{tps:.1f} tok/s incl. network), attempts={res.attempts}")
    print("READY")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ollama health check")
    ap.add_argument("--url", help="override EDA_OLLAMA_URL")
    ap.add_argument("--judge", action="store_true", help="also check the judge model")
    ap.add_argument("--no-speed", action="store_true", help="skip the 100-token speed test")
    args = ap.parse_args(argv)
    cfg = load_config(**({"ollama_url": args.url} if args.url else {}))
    ok = check_model(cfg, cfg.model, cfg.temperature, not args.no_speed)
    if args.judge:
        ok = check_model(cfg, cfg.judge_model, cfg.judge_temperature, not args.no_speed) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
