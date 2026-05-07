#!/usr/bin/env python3
"""Prototype a sub-context index layered outside SGLang's prefix cache.

This is a metadata-level prototype. It does not splice real KV cache blocks.
Instead, it consumes benchmark JSON emitted by bench_sglang_prefix_cache.py and
simulates the index decisions that a future runtime layer would make:

  hash(sub-context) -> token_range, kv_block_refs

The output is useful for comparing standard prefix reuse with segment-level
reuse opportunities when the same A/B/C sub-contexts are reordered.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SubContextEntry:
    content_hash: str
    subcontext_id: str
    source_request_name: str
    token_start: int
    token_end: int
    token_len: int
    kv_block_refs: list[str]


@dataclass(frozen=True)
class LookupDecision:
    request_name: str
    request_id: str
    order: str
    position: int
    subcontext_id: str
    content_hash: str
    token_start: int | None
    token_end: int | None
    token_len: int
    status: str
    reusable_tokens: int
    prefill_tokens: int
    source_request_name: str | None
    kv_block_refs: list[str]


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def make_kv_block_refs(row: dict[str, Any], block_size: int) -> list[str]:
    token_start = int(row["token_start"])
    token_end = int(row["token_end"])
    if token_end <= token_start:
        return []

    first_block = token_start // block_size
    last_block = (token_end - 1) // block_size
    return [
        (
            f"kvblk:{row['request_name']}:{row['subcontext_id']}:"
            f"{row['content_hash']}:b{block_id}"
        )
        for block_id in range(first_block, last_block + 1)
    ]


def valid_span(row: dict[str, Any]) -> bool:
    return all(row.get(key) not in (None, "") for key in ("token_start", "token_end", "token_len"))


def row_token_len(row: dict[str, Any]) -> int:
    if not valid_span(row):
        return 0
    return int(row["token_len"])


def request_rows_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["request_name"]: row for row in payload.get("requests", [])}


def metadata_by_request(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("subcontext_metadata", []):
        groups.setdefault(row["request_name"], []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row.get("position") or 0))
    return groups


class SubContextIndex:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.entries: dict[str, SubContextEntry] = {}

    def lookup(self, row: dict[str, Any]) -> SubContextEntry | None:
        return self.entries.get(row["content_hash"])

    def insert_if_missing(self, row: dict[str, Any]) -> None:
        if not valid_span(row):
            return
        content_hash = row["content_hash"]
        if content_hash in self.entries:
            return
        self.entries[content_hash] = SubContextEntry(
            content_hash=content_hash,
            subcontext_id=row["subcontext_id"],
            source_request_name=row["request_name"],
            token_start=int(row["token_start"]),
            token_end=int(row["token_end"]),
            token_len=int(row["token_len"]),
            kv_block_refs=make_kv_block_refs(row, self.block_size),
        )


def simulate(payload: dict[str, Any], block_size: int) -> tuple[list[LookupDecision], list[dict[str, Any]], dict[str, Any]]:
    req_rows = request_rows_by_name(payload)
    metadata_groups = metadata_by_request(payload)
    index = SubContextIndex(block_size=block_size)

    decisions: list[LookupDecision] = []
    summaries: list[dict[str, Any]] = []

    for req_row in payload.get("requests", []):
        request_name = req_row["request_name"]
        subcontext_rows = metadata_groups.get(request_name, [])
        request_decisions: list[LookupDecision] = []

        for row in subcontext_rows:
            token_len = row_token_len(row)
            hit = index.lookup(row)
            decision = LookupDecision(
                request_name=request_name,
                request_id=row["request_id"],
                order=row["order"],
                position=int(row.get("position") or 0),
                subcontext_id=row["subcontext_id"],
                content_hash=row["content_hash"],
                token_start=row.get("token_start"),
                token_end=row.get("token_end"),
                token_len=token_len,
                status="hit" if hit else "miss",
                reusable_tokens=token_len if hit else 0,
                prefill_tokens=0 if hit else token_len,
                source_request_name=hit.source_request_name if hit else None,
                kv_block_refs=hit.kv_block_refs if hit else [],
            )
            decisions.append(decision)
            request_decisions.append(decision)

        for row in subcontext_rows:
            index.insert_if_missing(row)

        total_tokens = sum(item.token_len for item in request_decisions)
        reusable_tokens = sum(item.reusable_tokens for item in request_decisions)
        prefill_tokens = sum(item.prefill_tokens for item in request_decisions)
        sglang_matched = req_row.get("lookup_matched_prefix_len")
        sglang_subcontext_overlap = (
            min(int(sglang_matched), total_tokens)
            if sglang_matched not in (None, "")
            else None
        )
        summaries.append(
            {
                "request_name": request_name,
                "subcontext_order": req_row.get("subcontext_order"),
                "subcontext_tokens": total_tokens,
                "prototype_reusable_tokens": reusable_tokens,
                "prototype_prefill_tokens": prefill_tokens,
                "prototype_hit_ratio": reusable_tokens / total_tokens if total_tokens else None,
                "sglang_prefix_matched_tokens": sglang_matched,
                "sglang_cached_tokens": req_row.get("log_cached_tokens"),
                "estimated_sglang_subcontext_overlap_tokens": sglang_subcontext_overlap,
                "additional_reuse_opportunity_tokens": (
                    max(reusable_tokens - sglang_subcontext_overlap, 0)
                    if sglang_subcontext_overlap is not None
                    else None
                ),
            }
        )

    index_dump = {
        content_hash: asdict(entry)
        for content_hash, entry in sorted(index.entries.items())
    }
    return decisions, summaries, index_dump


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = dict(row)
            if isinstance(normalized.get("kv_block_refs"), list):
                normalized["kv_block_refs"] = "|".join(normalized["kv_block_refs"])
            writer.writerow(normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_json", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for prototype CSV/JSON outputs. Defaults beside the input JSON.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Synthetic KV block size used only for readable kv_block_refs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.benchmark_json.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.benchmark_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions, summaries, index_dump = simulate(payload, args.block_size)

    stamp = now_stamp()
    base_name = f"subcontext_cache_prototype_{stamp}"
    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"

    write_csv(csv_path, [asdict(row) for row in decisions])
    json_path.write_text(
        json.dumps(
            {
                "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "source_benchmark_json": str(args.benchmark_json),
                "model": payload.get("model"),
                "block_size": args.block_size,
                "summaries": summaries,
                "decisions": [asdict(row) for row in decisions],
                "subcontext_index": index_dump,
                "note": (
                    "This prototype indexes sub-context metadata and marks reuse "
                    "opportunities. It does not prove that arbitrary KV blocks can "
                    "be safely spliced without runtime support for position and "
                    "attention-dependency handling."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Sub-context cache prototype")
    print("===========================")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    print("")
    print(
        "request, order, subcontext_tokens, prototype_reusable_tokens, "
        "prototype_hit_ratio, sglang_prefix_matched_tokens, extra_opportunity"
    )
    for row in summaries:
        hit_ratio = row["prototype_hit_ratio"]
        hit_ratio_text = "" if hit_ratio is None else f"{hit_ratio:.4f}"
        print(
            f"{row['request_name']}, {row['subcontext_order']}, "
            f"{row['subcontext_tokens']}, {row['prototype_reusable_tokens']}, "
            f"{hit_ratio_text}, {row['sglang_prefix_matched_tokens']}, "
            f"{row['additional_reuse_opportunity_tokens']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
