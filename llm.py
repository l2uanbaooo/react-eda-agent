"""Ollama HTTP client: health check, streaming chat, retry with backoff.

Mode A: cfg.ollama_url is the Kaggle tunnel (https://xxxx.trycloudflare.com).
Mode B: http://localhost:11434 inside the Kaggle notebook.

Why streaming: Cloudflare quick tunnels cut a request that sends no bytes for
100 s (HTTP 524). With stream=True Ollama sends tokens as they are produced,
and cfg.llm_read_timeout_s bounds the silence between chunks, not the total.

Retried (cfg.llm_max_retries times, waits base*2^k s): connection errors,
timeouts, a stream cut before done=true, HTTP 408/425/429/5xx and Cloudflare
52x/530 (tunnel down). Not retried: other 4xx (e.g. 404 model not pulled).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import requests

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 530}


class LLMError(RuntimeError):
    """Non-retryable LLM failure (bad request, model missing...)."""


class LLMUnavailable(LLMError):
    """The LLM could not be reached after all retries (tunnel down, server crash)."""


class _Retryable(Exception):
    pass


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_s: float
    attempts: int
    done_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class OllamaClient:
    def __init__(self, cfg, model: str | None = None, temperature: float | None = None,
                 session: requests.Session | None = None, sleep=time.sleep):
        self.cfg = cfg
        self.base = cfg.ollama_url.rstrip("/")
        self.model = model or cfg.model
        self.temperature = cfg.temperature if temperature is None else temperature
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "eda-agent/1.0"})
        self._sleep = sleep

    def options(self, num_predict: int | None = None) -> dict:
        return {"temperature": self.temperature, "seed": self.cfg.seed,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": num_predict if num_predict is not None else self.cfg.num_predict}

    @property
    def _timeout(self) -> tuple[float, float]:
        return (self.cfg.llm_connect_timeout_s, self.cfg.llm_read_timeout_s)

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list[dict], num_predict: int | None = None) -> ChatResult:
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "options": self.options(num_predict), "keep_alive": self.cfg.keep_alive}
        last = ""
        attempts = self.cfg.llm_max_retries + 1
        for attempt in range(attempts):
            try:
                return self._chat_once(payload, attempt + 1)
            except _Retryable as e:
                last = str(e)
                if attempt + 1 < attempts:
                    self._sleep(self.cfg.llm_backoff_base_s * (2 ** attempt))
        raise LLMUnavailable(f"LLM unreachable after {attempts} attempts at {self.base}: {last}")

    def _chat_once(self, payload: dict, attempt: int) -> ChatResult:
        t0 = time.perf_counter()
        try:
            r = self.session.post(f"{self.base}/api/chat", json=payload, stream=True,
                                  timeout=self._timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise _Retryable(f"{type(e).__name__}: {e}") from e
        with r:
            if r.status_code in RETRY_STATUS:
                raise _Retryable(f"HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code != 200:
                hint = " (model not pulled? run: ollama pull <model>)" if r.status_code == 404 else ""
                raise LLMError(f"HTTP {r.status_code}{hint}: {r.text[:300]}")
            parts, final = [], None
            try:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise _Retryable(f"non-JSON line from server (proxy page?): {line[:120]!r}") from e
                    if "error" in obj:
                        raise _Retryable(f"server error: {obj['error']}")
                    parts.append(obj.get("message", {}).get("content", ""))
                    if obj.get("done"):
                        final = obj
                        break
            except (requests.ConnectionError, requests.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                raise _Retryable(f"stream interrupted: {type(e).__name__}: {e}") from e
        if final is None:
            raise _Retryable("stream ended before done=true (connection cut?)")
        return ChatResult(text="".join(parts), model=final.get("model", self.model),
                          prompt_tokens=final.get("prompt_eval_count"),
                          completion_tokens=final.get("eval_count"),
                          duration_s=round(time.perf_counter() - t0, 3), attempts=attempt,
                          done_reason=final.get("done_reason"))

    # ---------------------------------------------------------------- health
    def health(self, warmup: bool = True) -> dict:
        """Check the server and the model BEFORE a run. Never raises."""
        out: dict = {"ok": False, "url": self.base, "model": self.model}
        try:
            v = self.session.get(f"{self.base}/api/version", timeout=self._timeout)
            v.raise_for_status()
            out["version"] = v.json().get("version")
            t = self.session.get(f"{self.base}/api/tags", timeout=self._timeout)
            t.raise_for_status()
            names = sorted({m.get("name") or m.get("model") for m in t.json().get("models", [])})
        except (requests.RequestException, ValueError) as e:
            out["error"] = f"server not reachable: {type(e).__name__}: {e}"
            return out
        out["models"] = names
        wanted = {self.model, f"{self.model}:latest"}
        out["model_present"] = bool(wanted & set(names))
        if not out["model_present"]:
            out["error"] = f"model {self.model!r} not pulled on the server (have: {names})"
            return out
        if warmup:
            try:
                res = self.chat([{"role": "user", "content": "Reply with the single word OK."}],
                                num_predict=5)
            except LLMError as e:
                out["error"] = f"warm-up chat failed: {e}"
                return out
            out["warmup_s"] = res.duration_s
            out["warmup_reply"] = res.text.strip()[:40]
        out["ok"] = True
        return out
