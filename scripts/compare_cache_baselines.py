#!/usr/bin/env python3
"""Merge No cache, RadixAttention, and sub-context prototype results."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cache_ratio(row: dict[str, Any]) -> float | None:
    direct = as_float(row.get("cached_token_ratio"))
    if direct is not None:
        return direct
    prompt_tokens = as_int(row.get("log_prompt_tokens") or row.get("usage_prompt_tokens"))
    cached_tokens = as_int(row.get("log_cached_tokens"))
    if prompt_tokens is None or prompt_tokens <= 0 or cached_tokens is None:
        return None
    return cached_tokens / prompt_tokens


def prefill_tokens(row: dict[str, Any]) -> int | None:
    direct = as_int(row.get("prefill_tokens_est"))
    if direct is not None:
        return direct
    prompt_tokens = as_int(row.get("log_prompt_tokens") or row.get("usage_prompt_tokens"))
    cached_tokens = as_int(row.get("log_cached_tokens"))
    if prompt_tokens is None or cached_tokens is None:
        return None
    return max(prompt_tokens - cached_tokens, 0)


def request_rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("requests", []):
        rows.append(
            {
                "group": group,
                "request_name": row.get("request_name"),
                "subcontext_order": row.get("subcontext_order"),
                "prompt_tokens": row.get("log_prompt_tokens") or row.get("usage_prompt_tokens"),
                "cached_tokens": row.get("log_cached_tokens"),
                "cached_token_ratio": cache_ratio(row),
                "prefill_tokens": prefill_tokens(row),
                "first_token_latency_s": row.get("first_token_latency_s")
                or row.get("log_time_to_prefill_finished_s")
                or row.get("log_prefill_launch_latency"),
                "total_latency_s": row.get("total_latency_s")
                or row.get("log_e2e_latency")
                or row.get("latency_s"),
                "lookup_matched_prefix_len": row.get("lookup_matched_prefix_len"),
                "latency_status": "measured",
            }
        )
    return rows


def proposed_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("summaries", []):
        rows.append(
            {
                "group": "Proposed_SubContextIndex",
                "request_name": row.get("request_name"),
                "subcontext_order": row.get("subcontext_order"),
                "prompt_tokens": row.get("prompt_tokens"),
                "cached_tokens": row.get("prototype_reusable_tokens"),
                "cached_token_ratio": row.get("cached_token_ratio"),
                "prefill_tokens": row.get("prefill_tokens_est"),
                "first_token_latency_s": row.get("first_token_latency_s"),
                "total_latency_s": row.get("total_latency_s"),
                "lookup_matched_prefix_len": row.get("sglang_prefix_matched_tokens"),
                "latency_status": row.get("latency_status", "not_measured_metadata_prototype"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "group",
        "request_name",
        "subcontext_order",
        "prompt_tokens",
        "cached_tokens",
        "cached_token_ratio",
        "prefill_tokens",
        "first_token_latency_s",
        "total_latency_s",
        "lookup_matched_prefix_len",
        "latency_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache-json", type=Path, required=True)
    parser.add_argument("--radix-json", type=Path, required=True)
    parser.add_argument("--proposed-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    no_cache = json.loads(args.no_cache_json.read_text(encoding="utf-8"))
    radix = json.loads(args.radix_json.read_text(encoding="utf-8"))
    proposed = json.loads(args.proposed_json.read_text(encoding="utf-8"))

    rows = []
    rows.extend(request_rows(no_cache, "Baseline_NoCache"))
    rows.extend(request_rows(radix, "Baseline_RadixAttention"))
    rows.extend(proposed_rows(proposed))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"single_vs_subcontext_cache_comparison_{stamp}.csv"
    json_path = args.output_dir / f"single_vs_subcontext_cache_comparison_{stamp}.json"
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "no_cache_json": str(args.no_cache_json),
                "radix_json": str(args.radix_json),
                "proposed_json": str(args.proposed_json),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Cache baseline comparison")
    print("=========================")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print("")
    print("group, request, ratio, prefill_tokens, first_token_latency_s, total_latency_s")
    for row in rows:
        ratio = row.get("cached_token_ratio")
        ratio_text = "" if ratio is None else f"{float(ratio):.4f}"
        print(
            f"{row['group']}, {row['request_name']}, {ratio_text}, "
            f"{row.get('prefill_tokens')}, {row.get('first_token_latency_s')}, "
            f"{row.get('total_latency_s')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
