#!/usr/bin/env python3
"""Tiny local OpenAI/SGLang-compatible mock server for runtime replay smoke tests.

This server is intentionally simple. It implements:

- `GET /health`
- `GET /metrics`
- `POST /v1/chat/completions`

It is not a model server and does not measure real SGLang KV-cache behavior.
Use it only to verify local replay plumbing before running against SGLang.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


STATE = {
    "cached_tokens_total": 0,
    "prompt_tokens_total": 0,
    "generation_tokens_total": 0,
    "num_requests_total": 0,
    "num_used_tokens": 0,
    "seen_prompts": set(),
    "last_latency_s": 0.0,
    "last_ttft_s": 0.0,
}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def prompt_from_payload(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    parts = []
    for message in messages:
        if isinstance(message, dict):
            parts.append(str(message.get("content", "")))
    return "\n".join(parts)


def metric_lines() -> str:
    requests = max(1, int(STATE["num_requests_total"]))
    prompt_total = float(STATE["prompt_tokens_total"])
    cached_total = float(STATE["cached_tokens_total"])
    new_ratio = 1.0 - (cached_total / prompt_total) if prompt_total else 1.0
    hit_rate = cached_total / prompt_total if prompt_total else 0.0
    return "\n".join(
        [
            f"sglang:cached_tokens_total {STATE['cached_tokens_total']}",
            f"sglang:prompt_tokens_total {STATE['prompt_tokens_total']}",
            f"sglang:generation_tokens_total {STATE['generation_tokens_total']}",
            f"sglang:num_requests_total {STATE['num_requests_total']}",
            f"sglang:cache_hit_rate {hit_rate}",
            f"sglang:new_token_ratio {new_ratio}",
            f"sglang:num_used_tokens {STATE['num_used_tokens']}",
            f"sglang:time_to_first_token_seconds_sum {STATE['last_ttft_s']}",
            f"sglang:e2e_request_latency_seconds_sum {STATE['last_latency_s']}",
            f"sglang:e2e_request_latency_seconds_count {requests}",
            "",
        ]
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "MockSGLang/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(fmt, *args)

    def send_json(self, data: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json({"status": "ok"})
            return
        if self.path == "/metrics":
            raw = metric_lines().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json({"error": "not found"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        prompt = prompt_from_payload(payload)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_tokens = estimate_tokens(prompt)
        completion = "Mock answer for local runtime replay smoke test."
        completion_tokens = estimate_tokens(completion)
        cached_tokens = prompt_tokens if prompt_hash in STATE["seen_prompts"] else 0
        STATE["seen_prompts"].add(prompt_hash)
        STATE["cached_tokens_total"] += cached_tokens
        STATE["prompt_tokens_total"] += prompt_tokens
        STATE["generation_tokens_total"] += completion_tokens
        STATE["num_requests_total"] += 1
        STATE["num_used_tokens"] += prompt_tokens + completion_tokens - cached_tokens
        STATE["last_ttft_s"] = 0.01 if cached_tokens else 0.02
        STATE["last_latency_s"] = 0.03 if cached_tokens else 0.05

        self.send_json(
            {
                "id": f"mock-{prompt_hash[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model", "mock-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": completion},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "cached_tokens": cached_tokens,
                },
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.quiet = args.quiet  # type: ignore[attr-defined]
    print(f"Mock OpenAI/SGLang server listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
