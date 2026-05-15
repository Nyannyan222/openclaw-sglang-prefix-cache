#!/usr/bin/env python3
"""Prepare a real WildClaw semantic sub-context order-permutation runtime test.

This experiment uses the same real WildClaw semantic sub-context set in two
orders. It is designed to show that native SGLang RadixAttention prefix cache
reuses identical ordered prefixes, but loses reuse when the same semantic
blocks appear in a different order.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SEMANTIC = Path("benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_semantic_subcontext_pilot.jsonl")
DEFAULT_RAW = Path("benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_real_contexts_pilot.jsonl")


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


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


def group_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    return dict(grouped)


def raw_prompt_by_task(rows: list[dict[str, Any]]) -> dict[str, str]:
    prompts = {}
    for row in rows:
        task = row.get("task", {})
        task_id = task.get("task_id")
        if task_id:
            prompts[task_id] = task.get("prompt", "")
    return prompts


def score_segment(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(row.get("relevance_score") or 0.0),
        float(row.get("completeness_score") or 0.0),
        int(row.get("sub_context_chars") or len(row.get("sub_context", ""))),
    )


def choose_segments(rows: list[dict[str, Any]], max_contexts: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=score_segment, reverse=True)
    chosen = ranked[:max_contexts]
    return sorted(chosen, key=lambda row: row.get("id", ""))


def permute_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) <= 2:
        return list(reversed(rows))
    return [rows[-1], *rows[1:-1], rows[0]]


def render_prompt(task_id: str, task_prompt: str, segments: list[dict[str, Any]], condition: str) -> str:
    blocks = []
    for index, row in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    (
                        f"[REAL_SEMANTIC_SUB_CONTEXT order={index} "
                        f"id={row.get('id')} hash={row.get('content_hash', '')[:16]}]"
                    ),
                    f"title: {row.get('title', '')}",
                    f"source_spans: {json.dumps(row.get('source_spans', []), ensure_ascii=False)}",
                    row.get("sub_context", ""),
                    "[/REAL_SEMANTIC_SUB_CONTEXT]",
                ]
            )
        )
    return "\n\n".join(
        [
            "You are answering a WildClawBench task using only the provided real semantic sub-context evidence.",
            "This prompt is part of an order-permutation cache limitation test.",
            f"condition: {condition}",
            f"task_id: {task_id}",
            "Original task prompt:",
            task_prompt,
            "Real semantic sub-context evidence:",
            "\n\n".join(blocks),
            "Answer briefly and cite which sub-context ids you used.",
        ]
    )


def write_protocol(path: Path, rows: list[dict[str, Any]], manifest_path: Path) -> None:
    task_count = len({row["task_id"] for row in rows})
    lines = [
        "# WildClaw Real Semantic Order-Permutation Test",
        "",
        "## Purpose",
        "",
        "Show the limitation of native SGLang RadixAttention prefix cache using real WildClaw semantic sub-contexts.",
        "",
        "The test sends, per task, three prompts:",
        "",
        "1. `real_semantic_order_original`: real semantic sub-contexts in one stable order.",
        "2. `real_semantic_order_repeat`: exact repeat of the original prompt, expected to show high prefix reuse.",
        "3. `real_semantic_order_permuted`: same real sub-context set in a different order, expected to show lower prefix reuse.",
        "",
        "## Runtime Command",
        "",
        "```bash",
        "sbatch --account=MST114180 --export=ALL,"
        f"MANIFEST={manifest_path.as_posix()},LIMIT=0,REPEAT=1,MAX_TOKENS=64,OUTPUT_DIR=benchmark_results/wildclaw_order_permutation_runtime_runs "
        "scripts/slurm_run_wildclaw_runtime_replay.sh",
        "```",
        "",
        "Use `REPEAT=1` because the manifest already includes the explicit repeat row.",
        "",
        "## Scope",
        "",
        f"- Tasks: {task_count}",
        f"- Manifest rows: {len(rows)}",
        "- Source: WildClawBench real task prompts and LLM semantic sub-context JSONL.",
        "",
        "## Interpretation",
        "",
        "The important comparison is within each task group:",
        "",
        "- Original -> repeat should have high cached-token ratio.",
        "- Original -> permuted should reuse only the shared prompt header and any unchanged leading text.",
        "- If the same semantic evidence appears in a different order but cache reuse drops, this supports the claim that prefix cache is order-sensitive.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-jsonl", type=Path, default=DEFAULT_SEMANTIC)
    parser.add_argument("--raw-contexts", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/wildclaw_order_permutation_test"))
    parser.add_argument("--max-tasks", type=int, default=3)
    parser.add_argument("--min-contexts", type=int, default=2)
    parser.add_argument("--max-contexts", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = args.output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    semantic_by_task = group_by_task(read_jsonl(args.semantic_jsonl))
    prompts = raw_prompt_by_task(read_jsonl(args.raw_contexts))
    selected_tasks = [
        task_id
        for task_id, rows in sorted(semantic_by_task.items())
        if len(rows) >= args.min_contexts and task_id in prompts
    ][: args.max_tasks]

    manifest = []
    summary = []
    for task_id in selected_tasks:
        original_segments = choose_segments(semantic_by_task[task_id], args.max_contexts)
        permuted_segments = permute_segments(original_segments)
        variants = [
            ("real_semantic_order_original", original_segments),
            ("real_semantic_order_repeat", original_segments),
            ("real_semantic_order_permuted", permuted_segments),
        ]
        group_id = f"{task_id}::real_semantic_order_permutation"
        segment_ids = [row["id"] for row in original_segments]
        for condition, segments in variants:
            prompt_path = prompts_dir / f"{task_id}__{condition}.txt"
            prompt_path.write_text(render_prompt(task_id, prompts[task_id], segments, condition), encoding="utf-8")
            manifest.append(
                {
                    "id": f"{group_id}::{condition}",
                    "task_id": task_id,
                    "condition": condition,
                    "prompt_path": prompt_path.as_posix(),
                    "source_semantic_jsonl": args.semantic_jsonl.as_posix(),
                    "segment_ids": [row["id"] for row in segments],
                    "segment_hashes": [row.get("content_hash", "") for row in segments],
                    "notes": "Real WildClaw semantic sub-context order-permutation cache limitation row.",
                }
            )
        summary.append(
            {
                "task_id": task_id,
                "segment_count": len(original_segments),
                "original_segment_ids": " | ".join(segment_ids),
                "permuted_segment_ids": " | ".join(row["id"] for row in permuted_segments),
            }
        )

    manifest_path = args.output_dir / "wildclaw_order_permutation_runtime_manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    write_csv(args.output_dir / "wildclaw_order_permutation_summary.csv", summary)
    (args.output_dir / "wildclaw_order_permutation_protocol.json").write_text(
        json.dumps(
            {
                "created_at": now_iso(),
                "semantic_jsonl": args.semantic_jsonl.as_posix(),
                "raw_contexts": args.raw_contexts.as_posix(),
                "manifest": manifest,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_protocol(args.output_dir / "wildclaw_order_permutation_protocol.md", manifest, manifest_path)

    print("WildClaw real semantic order-permutation test")
    print("============================================")
    print(f"Output dir: {args.output_dir}")
    print(f"Tasks: {len(selected_tasks)}")
    print(f"Manifest rows: {len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
