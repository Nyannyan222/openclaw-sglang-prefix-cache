#!/usr/bin/env python3
"""Prepare a runtime protocol for semantic non-prefix KV reuse experiments.

This script does not splice KV tensors. It creates SGLang replay prompts and a
semantic block manifest where identical sub-context blocks appear in a different
order. Native prefix cache should only reuse the shared prefix, while a future
runtime-level semantic KV layer could reuse the repeated block hashes even when
their prompt positions change.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        writer.writerows(rows)


def group_segments(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: row.get("id", ""))
    return dict(grouped)


def prompt_for(task_id: str, variant: str, segments: list[dict[str, Any]]) -> str:
    blocks = []
    for index, row in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[SEMANTIC_BLOCK index={index} id={row.get('id')} hash={content_hash(row.get('sub_context', ''))[:16]}]",
                    row.get("sub_context", ""),
                    "[/SEMANTIC_BLOCK]",
                ]
            )
        )
    return "\n\n".join(
        [
            "Answer the WildClawBench task using the semantic blocks below.",
            "This prompt is part of a non-prefix KV reuse experiment.",
            f"task_id: {task_id}",
            f"variant: {variant}",
            "[SEMANTIC_CONTEXT]",
            "\n\n".join(blocks),
            "[/SEMANTIC_CONTEXT]",
            "Return a concise answer and cite semantic block ids.",
        ]
    )


def rotate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) <= 1:
        return rows
    mid = max(1, len(rows) // 2)
    return rows[mid:] + rows[:mid]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/semantic_nonprefix_kv_reuse"))
    parser.add_argument("--max-segments-per-task", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=0, help="0 means all tasks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt_dir = args.output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    groups = group_segments(read_jsonl(args.semantic_jsonl))
    selected_task_ids = sorted(groups)
    if args.max_tasks:
        selected_task_ids = selected_task_ids[: args.max_tasks]

    runtime_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for task_id in selected_task_ids:
        original = groups[task_id][: args.max_segments_per_task]
        if len(original) < 2:
            continue
        variants = {
            "semantic_order_a": original,
            "semantic_order_b_nonprefix": rotate(original),
            "semantic_order_a_repeat": original,
        }
        for variant, segments in variants.items():
            prompt = prompt_for(task_id, variant, segments)
            prompt_path = prompt_dir / f"{safe_name(task_id)}__{variant}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            runtime_rows.append(
                {
                    "id": f"{task_id}::{variant}",
                    "task_id": task_id,
                    "condition": variant,
                    "prompt_path": prompt_path.as_posix(),
                    "kv_reuse_mode": (
                        "native_prefix_cache_observation"
                        if variant != "semantic_order_b_nonprefix"
                        else "semantic_subcontext_kv_reuse_candidate"
                    ),
                    "notes": "Order B reuses the same semantic block hashes in non-prefix positions.",
                }
            )
            for position, segment in enumerate(segments, start=1):
                text = segment.get("sub_context", "")
                block_rows.append(
                    {
                        "runtime_row_id": f"{task_id}::{variant}",
                        "task_id": task_id,
                        "variant": variant,
                        "position": position,
                        "subcontext_id": segment.get("id", ""),
                        "content_hash": content_hash(text),
                        "subcontext_chars": len(text),
                        "expected_runtime_action": (
                            "lookup_semantic_kv_block_by_hash_and_remap_position"
                            if variant == "semantic_order_b_nonprefix"
                            else "insert_or_prefix_lookup"
                        ),
                    }
                )
        summary_rows.append(
            {
                "task_id": task_id,
                "semantic_blocks": len(original),
                "order_a": " -> ".join(row.get("id", "") for row in original),
                "order_b": " -> ".join(row.get("id", "") for row in rotate(original)),
                "nonprefix_reuse_target": "same content_hashes, different positions",
            }
        )

    write_jsonl(args.output_dir / "semantic_nonprefix_runtime_manifest.jsonl", runtime_rows)
    write_jsonl(args.output_dir / "semantic_kv_block_manifest.jsonl", block_rows)
    write_csv(args.output_dir / "semantic_nonprefix_task_summary.csv", summary_rows)
    (args.output_dir / "semantic_nonprefix_protocol.json").write_text(
        json.dumps(
            {
                "created_at": now_iso(),
                "semantic_jsonl": args.semantic_jsonl.as_posix(),
                "runtime_manifest": (args.output_dir / "semantic_nonprefix_runtime_manifest.jsonl").as_posix(),
                "block_manifest": (args.output_dir / "semantic_kv_block_manifest.jsonl").as_posix(),
                "runtime_patch_target": {
                    "lookup_key": "content_hash",
                    "required_sglang_hook": "before prefill allocation, after tokenization and semantic span alignment",
                    "position_policy": "reuse only blocks whose internal tokenization matches; remap RoPE positions or restrict to position-invariant validation first",
                    "safety_policy": "never reuse across model, tokenizer, system prompt, sampling-affecting adapter, or tenant boundary",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Semantic non-prefix KV reuse protocol prepared")
    print("==============================================")
    print(f"Output dir: {args.output_dir}")
    print(f"Runtime rows: {len(runtime_rows)}")
    print(f"Semantic block rows: {len(block_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
