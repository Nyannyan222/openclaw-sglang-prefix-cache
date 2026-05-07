#!/usr/bin/env python3
"""Benchmark SGLang RadixAttention prefix-cache behavior for R1/R2/R3 prompts.

The script sends:
  R1 = A + B + C + question1
  R2 = A + B + C + question2
  R3 = C + A + B + question3

It records OpenAI-compatible response usage, Prometheus metric deltas, per-request
cached token counts from SGLang logs, and sub-context span metadata for A/B/C.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


METRICS_OF_INTEREST = (
    "sglang:cached_tokens_total",
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:num_requests_total",
    "sglang:cache_hit_rate",
    "sglang:new_token_ratio",
    "sglang:num_used_tokens",
    "sglang:realtime_tokens_total",
)


def now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def derive_metrics_url(base_url: str) -> str:
    base_url = normalize_base_url(base_url)
    if base_url.endswith("/v1"):
        return base_url[:-3] + "/metrics"
    return base_url + "/metrics"


def http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
    api_key: str | None = None,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {raw[:500]}") from exc


def http_text(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Return summed metric values by metric name."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name_with_labels, value_text = parts
        name = name_with_labels.split("{", 1)[0]
        try:
            value = float(value_text)
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + value
    return values


def scrape_metrics(metrics_url: str, timeout: float) -> dict[str, float]:
    return parse_prometheus_metrics(http_text(metrics_url, timeout=timeout))


def metric_delta(
    before: dict[str, float],
    after: dict[str, float],
    metric_name: str,
    default_when_absent: float | None = None,
) -> float | None:
    if metric_name not in before and metric_name not in after:
        return default_when_absent
    return after.get(metric_name, 0.0) - before.get(metric_name, 0.0)


def metric_value(
    metrics: dict[str, float], metric_name: str, default_when_absent: float | None = None
) -> float | None:
    if metric_name not in metrics:
        return default_when_absent
    return metrics[metric_name]


def is_counter_metric(metric_name: str) -> bool:
    return metric_name.endswith("_total")


def metric_default(metric_name: str) -> float | None:
    if is_counter_metric(metric_name):
        return 0.0
    if metric_name in {"sglang:cache_hit_rate", "sglang:new_token_ratio"}:
        return 0.0
    return None


def safe_get(mapping: Any, path: list[str | int], default: Any = None) -> Any:
    cur: Any = mapping
    for part in path:
        if isinstance(part, int):
            if not isinstance(cur, list) or part >= len(cur):
                return default
            cur = cur[part]
            continue
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_subcontexts(experiment_id: str) -> dict[str, str]:
    return {
        "A": f"""
[SUBCONTEXT A | experiment={experiment_id}]
Product Alpha is a quiet workstation laptop for data science students.
Alpha has 32 GB memory, a 14 inch display, and excellent keyboard travel.
Alpha's strongest point is local Python development and light CUDA testing.
Alpha's limitation is its small battery under sustained GPU load.
Alpha should be recommended when the user values portability and coding comfort.
""".strip(),
        "B": f"""
[SUBCONTEXT B | experiment={experiment_id}]
Product Beta is a compact inference server for lab experiments.
Beta has 96 GB memory, dual network ports, and room for a full-size GPU.
Beta's strongest point is serving local language models for several users.
Beta's limitation is noise, heat, and higher idle power consumption.
Beta should be recommended when the user values throughput and shared access.
""".strip(),
        "C": f"""
[SUBCONTEXT C | experiment={experiment_id}]
Product Gamma is a low-cost mini PC used as an always-on automation node.
Gamma has 16 GB memory, integrated graphics, and very low idle power.
Gamma's strongest point is running schedulers, scripts, and dashboards.
Gamma's limitation is weak model inference performance.
Gamma should be recommended when the user values low cost and reliability.
""".strip(),
    }


def build_prompt(
    experiment_id: str, order: list[str], subcontexts: dict[str, str], question: str
) -> str:
    intro = f"""
Experiment id: {experiment_id}
You are running a cache benchmark for SGLang.
Use only the supplied subcontexts. Keep the answer to two short sentences.
""".strip()
    context_block = "\n\n".join(subcontexts[key] for key in order)
    return f"{intro}\n\n{context_block}\n\n[QUESTION]\n{question}"


def build_requests(experiment_id: str) -> list[dict[str, Any]]:
    subcontexts = make_subcontexts(experiment_id)
    specs = [
        (
            "R1_ABC",
            ["A", "B", "C"],
            "Which product best fits a portable student developer, and why?",
        ),
        (
            "R2_ABC_same_prefix",
            ["A", "B", "C"],
            "Which product best fits a shared local inference lab, and why?",
        ),
        (
            "R3_CAB_reordered",
            ["C", "A", "B"],
            "Which product best fits an always-on automation node, and why?",
        ),
    ]
    rows = []
    for name, order, question in specs:
        prompt = build_prompt(experiment_id, order, subcontexts, question)
        rows.append(
            {
                "name": name,
                "order": "+".join(order),
                "order_compact": "".join(order),
                "order_ids": order,
                "question": question,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "prompt_sha256": prompt_sha256(prompt),
                "subcontexts": subcontexts,
            }
        )
    return rows


def load_tokenizer(tokenizer_path: str) -> Any | None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("WARNING: transformers is not installed; sub-context token spans will be empty.")
        return None
    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, use_fast=True)
    except Exception as exc:
        print(
            f"WARNING: failed to load tokenizer '{tokenizer_path}': {exc}. "
            "Sub-context token spans will be empty."
        )
        return None


def prompt_token_offsets(tokenizer: Any | None, prompt: str) -> tuple[int | None, list[tuple[int, int]]]:
    if tokenizer is None:
        return None, []
    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return None, []

    input_ids = encoded.get("input_ids", [])
    offsets = encoded.get("offset_mapping", [])
    normalized_offsets: list[tuple[int, int]] = []
    for offset in offsets:
        if isinstance(offset, tuple):
            normalized_offsets.append(offset)
        else:
            normalized_offsets.append(tuple(offset))
    return len(input_ids), normalized_offsets


def token_range_for_char_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int | None, int | None, int | None]:
    token_indices = [
        idx
        for idx, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > char_start and tok_start < char_end and tok_end > tok_start
    ]
    if not token_indices:
        return None, None, None
    token_start = token_indices[0]
    token_end = token_indices[-1] + 1
    return token_start, token_end, token_end - token_start


def build_subcontext_metadata_for_request(
    req_spec: dict[str, Any],
    *,
    tokenizer: Any | None,
) -> list[dict[str, Any]]:
    prompt = req_spec["prompt"]
    tokenizer_prompt_tokens, offsets = prompt_token_offsets(tokenizer, prompt)
    rows: list[dict[str, Any]] = []

    for position, subcontext_id in enumerate(req_spec["order_ids"], start=1):
        content = req_spec["subcontexts"][subcontext_id]
        char_start = prompt.index(content)
        char_end = char_start + len(content)
        token_start, token_end, token_len = token_range_for_char_span(
            offsets, char_start, char_end
        )
        rows.append(
            {
                "request_id": req_spec["name"],
                "subcontext_id": subcontext_id,
                "char_start": char_start,
                "char_end": char_end,
                "token_start": token_start,
                "token_end": token_end,
                "token_len": token_len,
                "content_hash": content_sha256(content)[:16],
                "order": req_spec["order_compact"],
                "position": position,
                "tokenizer_prompt_tokens": tokenizer_prompt_tokens,
            }
        )
    return rows


def request_cache_fields(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    prompt_tokens = row.get("log_prompt_tokens") or row.get("usage_prompt_tokens")
    cached_tokens = row.get("log_cached_tokens")
    if cached_tokens is None:
        cached_tokens = row.get("metric_delta_cached_tokens_total")

    cache_ratio = None
    try:
        if prompt_tokens is not None and float(prompt_tokens) > 0 and cached_tokens is not None:
            cache_ratio = float(cached_tokens) / float(prompt_tokens)
    except (TypeError, ValueError):
        cache_ratio = None
    return cached_tokens, prompt_tokens, cache_ratio


def attach_request_cache_to_subcontexts(
    subcontext_rows: list[dict[str, Any]], request_row: dict[str, Any]
) -> list[dict[str, Any]]:
    cached_tokens, prompt_tokens, cache_ratio = request_cache_fields(request_row)
    return [
        {
            **subcontext_row,
            "cached_tokens": cached_tokens,
            "prompt_tokens": prompt_tokens,
            "cache_ratio": cache_ratio,
        }
        for subcontext_row in subcontext_rows
    ]


def find_finished_log_event(log_file: str | None, request_id: str) -> dict[str, Any] | None:
    if not log_file:
        return None
    path = pathlib.Path(log_file)
    if not path.exists():
        return None

    found = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{") or request_id not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "request.finished" and event.get("rid") == request_id:
                    found = event
    except OSError:
        return None
    return found


def run_one_request(
    *,
    req_spec: dict[str, Any],
    chat_url: str,
    metrics_url: str,
    model: str,
    api_key: str | None,
    timeout: float,
    max_tokens: int,
    temperature: float,
    log_file: str | None,
    sleep_after_request: float,
) -> dict[str, Any]:
    before_metrics = scrape_metrics(metrics_url, timeout=min(timeout, 20.0))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": req_spec["prompt"]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    started = time.perf_counter()
    response = http_json(chat_url, payload=payload, timeout=timeout, api_key=api_key)
    latency_s = time.perf_counter() - started

    if sleep_after_request:
        time.sleep(sleep_after_request)
    after_metrics = scrape_metrics(metrics_url, timeout=min(timeout, 20.0))

    request_id = response.get("id", "")
    log_event = find_finished_log_event(log_file, request_id)
    meta_info = safe_get(log_event or {}, ["out", "meta_info"], {})
    usage = response.get("usage") or {}

    row: dict[str, Any] = {
        "request_name": req_spec["name"],
        "subcontext_order": req_spec["order"],
        "question": req_spec["question"],
        "prompt_chars": req_spec["prompt_chars"],
        "prompt_sha256": req_spec["prompt_sha256"],
        "response_id": request_id,
        "latency_s": round(latency_s, 6),
        "response_text": safe_get(response, ["choices", 0, "message", "content"], ""),
        "usage_prompt_tokens": usage.get("prompt_tokens"),
        "usage_completion_tokens": usage.get("completion_tokens"),
        "usage_total_tokens": usage.get("total_tokens"),
        "log_prompt_tokens": meta_info.get("prompt_tokens"),
        "log_completion_tokens": meta_info.get("completion_tokens"),
        "log_cached_tokens": meta_info.get("cached_tokens"),
        "log_e2e_latency": meta_info.get("e2e_latency"),
        "log_prefill_launch_latency": meta_info.get("prefill_launch_latency"),
        "log_queue_time": meta_info.get("queue_time"),
    }

    for metric_name in METRICS_OF_INTEREST:
        short_name = metric_name.replace("sglang:", "").replace("-", "_")
        default = metric_default(metric_name)
        row[f"metric_delta_{short_name}"] = metric_delta(
            before_metrics, after_metrics, metric_name, default_when_absent=default
        )
        row[f"metric_after_{short_name}"] = metric_value(
            after_metrics, metric_name, default_when_absent=default
        )

    row["_raw_response"] = response
    row["_raw_log_event"] = log_event
    return row


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    hidden_prefix = "_"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith(hidden_prefix):
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def print_summary(rows: list[dict[str, Any]], json_path: pathlib.Path, csv_path: pathlib.Path) -> None:
    print("\nBenchmark summary")
    print("=================")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    print("")
    print(
        "request, order, prompt_tokens, cached_tokens(log), "
        "cached_tokens_delta(metrics), latency_s"
    )
    for row in rows:
        print(
            f"{row['request_name']}, {row['subcontext_order']}, "
            f"{row.get('usage_prompt_tokens')}, {row.get('log_cached_tokens')}, "
            f"{row.get('metric_delta_cached_tokens_total')}, {row.get('latency_s')}"
        )


def print_subcontext_summary(rows: list[dict[str, Any]], csv_path: pathlib.Path) -> None:
    print("")
    print(f"Sub-context metadata CSV: {csv_path}")
    print("request, subcontext, order, char_span, token_span, token_len, cache_ratio")
    for row in rows:
        token_span = f"{row.get('token_start')}:{row.get('token_end')}"
        char_span = f"{row.get('char_start')}:{row.get('char_end')}"
        cache_ratio = row.get("cache_ratio")
        cache_ratio_text = f"{cache_ratio:.4f}" if isinstance(cache_ratio, float) else str(cache_ratio)
        print(
            f"{row['request_id']}, {row['subcontext_id']}, {row['order']}, "
            f"{char_span}, {token_span}, {row.get('token_len')}, {cache_ratio_text}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run R1/R2/R3 SGLang prefix-cache benchmark and write CSV/JSON."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--metrics-url", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path for sub-context token spans. Defaults to --model.",
    )
    parser.add_argument("--api-key", default=os.environ.get("SGLANG_API_KEY"))
    parser.add_argument("--log-file", default="/tmp/sglang_openclaw.log")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sleep-after-request", type=float, default=0.25)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional id embedded into prompts. Defaults to a fresh UUID.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    base_url = normalize_base_url(args.base_url)
    chat_url = base_url + "/chat/completions"
    metrics_url = args.metrics_url or derive_metrics_url(base_url)
    experiment_id = args.experiment_id or f"bench-{uuid.uuid4().hex[:12]}"

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"sglang_prefix_cache_{stamp}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    subcontext_csv_path = output_dir / f"{stem}_subcontexts.csv"

    tokenizer_path = args.tokenizer_path or args.model
    tokenizer = load_tokenizer(tokenizer_path)
    request_specs = build_requests(experiment_id)
    rows = []
    subcontext_rows = []
    for req_spec in request_specs:
        print(f"Running {req_spec['name']} ({req_spec['order']})...")
        request_row = run_one_request(
            req_spec=req_spec,
            chat_url=chat_url,
            metrics_url=metrics_url,
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            log_file=args.log_file,
            sleep_after_request=args.sleep_after_request,
        )
        rows.append(request_row)
        request_subcontext_rows = build_subcontext_metadata_for_request(
            req_spec,
            tokenizer=tokenizer,
        )
        subcontext_rows.extend(
            attach_request_cache_to_subcontexts(request_subcontext_rows, request_row)
        )

    payload = {
        "created_at": now_utc_iso(),
        "experiment_id": experiment_id,
        "base_url": base_url,
        "metrics_url": metrics_url,
        "model": args.model,
        "tokenizer_path": tokenizer_path,
        "tokenizer_loaded": tokenizer is not None,
        "log_file": args.log_file,
        "requests": rows,
        "subcontext_metadata": subcontext_rows,
    }
    write_json(json_path, payload)
    write_csv(csv_path, rows)
    write_csv(subcontext_csv_path, subcontext_rows)
    print_summary(rows, json_path, csv_path)
    print_subcontext_summary(subcontext_rows, subcontext_csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
