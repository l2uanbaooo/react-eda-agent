"""GD3: llm.py against a FAKE Ollama server (no GPU, no network needed).

Covers: streaming assembly, request payload (seed/temperature/num_ctx/keep_alive),
retry+backoff on 5xx/52x/530, cut streams, error lines, proxy HTML, read
timeout, connection refused, no retry on 404, health check, health_check.py.
"""
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from config import load_config
from llm import LLMError, LLMUnavailable, OllamaClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
import health_check  # noqa: E402

MODEL = "qwen2.5-coder:14b"


class FakeOllama:
    def __init__(self):
        self.script: list = []
        self.calls = 0
        self.bodies: list = []
        self.models = [MODEL]
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/version":
                    self._json(200, {"version": "0.5.7"})
                elif self.path == "/api/tags":
                    self._json(200, {"models": [{"name": m} for m in fake.models]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                fake.bodies.append(json.loads(self.rfile.read(n)))
                fake.calls += 1
                kind, arg = fake.script.pop(0) if fake.script else ("ok", "hello world")
                if kind == "status":
                    self.send_response(arg)
                    self.end_headers()
                    self.wfile.write(b"upstream error")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                if kind == "sleep":
                    time.sleep(arg)
                    return
                if kind == "html":
                    self.wfile.write(b"<html>tunnel error</html>\n")
                    return
                if kind == "error_line":
                    self.wfile.write(json.dumps({"error": arg}).encode() + b"\n")
                    return
                words = arg.split(" ")
                for i, w in enumerate(words):
                    piece = w if i == 0 else " " + w
                    self.wfile.write(json.dumps({"message": {"content": piece}, "done": False}).encode() + b"\n")
                    self.wfile.flush()
                if kind == "ok":
                    self.wfile.write(json.dumps({"done": True, "model": MODEL, "eval_count": len(words),
                                                 "prompt_eval_count": 7, "done_reason": "stop"}).encode() + b"\n")
                # kind == "cut": no final done line, connection closes

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake():
    f = FakeOllama()
    yield f
    f.close()


def _client(url, sleeps=None, **over):
    cfg = load_config(env={}, ollama_url=url, llm_read_timeout_s=over.pop("read", 5.0), **over)
    return OllamaClient(cfg, sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


def test_stream_assembled_and_payload(fake):
    res = _client(fake.url).chat([{"role": "user", "content": "hi"}])
    assert res.text == "hello world" and res.completion_tokens == 2 and res.prompt_tokens == 7
    assert res.attempts == 1 and res.done_reason == "stop"
    body = fake.bodies[0]
    assert body["stream"] is True and body["model"] == MODEL and body["keep_alive"] == "24h"
    assert body["options"] == {"temperature": 0.1, "seed": 42, "num_ctx": 8192, "num_predict": 1024}


@pytest.mark.parametrize("bad", [("status", 500), ("status", 502), ("status", 524), ("status", 530),
                                 ("cut", "half an answer"), ("error_line", "model runner crashed"),
                                 ("html", None)])
def test_retry_then_success(fake, bad):
    fake.script = [bad, ("ok", "fine")]
    sleeps = []
    res = _client(fake.url, sleeps).chat([{"role": "user", "content": "x"}])
    assert res.text == "fine" and res.attempts == 2 and fake.calls == 2
    assert sleeps == [1.0]


def test_backoff_then_give_up(fake):
    fake.script = [("status", 500)] * 10
    sleeps = []
    with pytest.raises(LLMUnavailable, match="after 5 attempts"):
        _client(fake.url, sleeps).chat([{"role": "user", "content": "x"}])
    assert fake.calls == 5 and sleeps == [1.0, 2.0, 4.0, 8.0]


def test_404_is_not_retried(fake):
    fake.script = [("status", 404)]
    with pytest.raises(LLMError, match="HTTP 404.*ollama pull") as ei:
        _client(fake.url).chat([{"role": "user", "content": "x"}])
    assert not isinstance(ei.value, LLMUnavailable) and fake.calls == 1


def test_read_timeout_is_retried(fake):
    fake.script = [("sleep", 1.5), ("ok", "late but ok")]
    res = _client(fake.url, read=0.5).chat([{"role": "user", "content": "x"}])
    assert res.text == "late but ok" and res.attempts == 2


def test_connection_refused():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed port after this block
    sleeps = []
    with pytest.raises(LLMUnavailable):
        _client(f"http://127.0.0.1:{port}", sleeps, llm_max_retries=2).chat([{"role": "user", "content": "x"}])
    assert sleeps == [1.0, 2.0]


def test_health_ok_and_model_missing(fake):
    h = _client(fake.url).health()
    assert h["ok"] and h["version"] == "0.5.7" and h["model_present"] and h["warmup_reply"]
    fake.models = ["llama3:8b"]
    h = _client(fake.url).health()
    assert not h["ok"] and "not pulled" in h["error"]


def test_health_unreachable_never_raises():
    h = _client("http://127.0.0.1:9").health()
    assert not h["ok"] and "not reachable" in h["error"]


def test_health_check_script_exit_codes(fake, capsys):
    assert health_check.main(["--url", fake.url]) == 0
    assert "READY" in capsys.readouterr().out
    fake.models = ["other:1b"]
    assert health_check.main(["--url", fake.url, "--no-speed"]) == 1
    assert "NOT READY" in capsys.readouterr().out
