"""Shared client helpers for the GB10 benchmark scripts.

Endpoint and key come from the environment so nothing is baked in:

    export GB10_BASE_URL=http://127.0.0.1:8000/v1   # SparkStation gateway
    export GB10_API_KEY=your-litellm-master-key
    export GB10_MODEL=default

For the standalone SGLang server (no SparkStation), use port 8888 and the
served model name instead:

    export GB10_BASE_URL=http://127.0.0.1:8888/v1
    export GB10_MODEL=qwen3.8-27b-sglang
"""
import json
import os
import time
import urllib.request

BASE_URL = os.environ.get("GB10_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
API_KEY = os.environ.get("GB10_API_KEY", "dummy-key")
MODEL = os.environ.get("GB10_MODEL", "default")

CHAT_URL = f"{BASE_URL}/chat/completions"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

# The canonical decode-throughput prompt used for every figure in the README.
CODE_PROMPT = (
    "Write a complete Python implementation of an LRUCache class with get and put "
    "in O(1), using a dict and a doubly linked list. Include docstrings."
)


def chat(prompt, max_tokens, thinking=False, stream=False, timeout=3600):
    """One chat completion.

    Returns a dict with e2e seconds, completion/prompt token counts, and (when
    streaming) ttft. Always count `completion_tokens` over wall time rather than
    counting SSE events: DFlash2 emits ~3.75 tokens per event, so event-counting
    inflates the rate by roughly 4x.
    """
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(CHAT_URL, json.dumps(body).encode(), HEADERS)
    t0 = time.time()

    if not stream:
        d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        usage = d["usage"]
        return {
            "e2e": time.time() - t0,
            "ttft": None,
            "completion_tokens": usage["completion_tokens"],
            "prompt_tokens": usage["prompt_tokens"],
            "finish_reason": d["choices"][0].get("finish_reason"),
            "content": d["choices"][0]["message"].get("content") or "",
        }

    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            choices = d.get("choices") or []
            if choices and ttft is None:
                delta = choices[0].get("delta", {}) or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    ttft = time.time() - t0
            if d.get("usage"):
                completion_tokens = d["usage"].get("completion_tokens", completion_tokens)
                prompt_tokens = d["usage"].get("prompt_tokens", prompt_tokens)
    return {
        "e2e": time.time() - t0,
        "ttft": ttft,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": None,
        "content": "",
    }
