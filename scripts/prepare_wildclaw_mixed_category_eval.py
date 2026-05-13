#!/usr/bin/env python3
"""Build a mixed-category WildClawBench evaluation manifest.

The manifest intentionally reuses the existing framework-eval runner shape:
each row contains an original task prompt, a context policy, context text, and
enough metadata to stage prompts or replay them through SGLang later.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TASKS = [
    "01_Productivity_Flow_task_1_arxiv_digest",
    "01_Productivity_Flow_task_6_calendar_scheduling",
    "04_Search_Retrieval_task_2_conflicting_handling",
    "04_Search_Retrieval_task_4_efficient_search",
    "04_Search_Retrieval_task_5_fuzzy_search",
    "06_Safety_Alignment_task_2_leaked_api",
    "06_Safety_Alignment_task_6_prompt_injection",
]


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


def concat_segments(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        parts.append(
            "\n".join(
                [
                    f"[SUB_CONTEXT id={row.get('id')} method={row.get('method')} title={row.get('title')}]",
                    row.get("sub_context", ""),
                    "[/SUB_CONTEXT]",
                ]
            )
        )
    return "\n\n".join(parts)


def task_prompt(raw: dict[str, Any]) -> str:
    return raw["task"].get("prompt", "")


def make_eval_prompt(raw: dict[str, Any], condition: str) -> str:
    task = raw["task"]
    return "\n\n".join(
        [
            "Use the provided context according to the context_policy.",
            "Do not use hidden grading answers. Complete the task objective and report the evidence you used.",
            f"context_policy: {condition}",
            f"task_id: {task['task_id']}",
            f"category: {task['category']}",
            f"task_name: {task.get('name', '')}",
            "Original task prompt:",
            task.get("prompt", ""),
        ]
    )


def manifest_row(raw: dict[str, Any], condition: str, chunks: list[dict[str, Any]], context_text: str) -> dict[str, Any]:
    task = raw["task"]
    full_chars = int(raw.get("long_context_chars") or len(raw.get("long_context", "")) or 1)
    context_chars = len(context_text)
    return {
        "id": f"{task['task_id']}::{condition}",
        "task_id": task["task_id"],
        "category": task["category"],
        "task_name": task.get("name", ""),
        "condition": condition,
        "chunk_count": len(chunks),
        "context_chars": context_chars,
        "full_context_chars": full_chars,
        "context_reduction": 1.0 - (context_chars / full_chars) if full_chars else 0.0,
        "workspace_missing": bool(raw.get("workspace_missing")),
        "workspace_dir": raw.get("workspace_dir", ""),
        "context_text": context_text,
        "eval_prompt": make_eval_prompt(raw, condition),
        "expected_output": "Framework answer plus evidence trace.",
        "notes": (
            "Prompt-only row because workspace is missing locally."
            if raw.get("workspace_missing")
            else "Real task prompt plus local workspace-derived context."
        ),
    }


def summarize(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        grouped[(row["category"], row["condition"])].append(row)
    summary = []
    for (category, condition), rows in sorted(grouped.items()):
        summary.append(
            {
                "category": category,
                "condition": condition,
                "task_count": len(rows),
                "workspace_missing_count": sum(1 for row in rows if row["workspace_missing"]),
                "avg_context_chars": round(sum(row["context_chars"] for row in rows) / len(rows), 1),
                "avg_context_reduction": round(sum(row["context_reduction"] for row in rows) / len(rows), 4),
            }
        )
    return summary


def write_plan(path: Path, manifest: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    categories = sorted({row["category"] for row in manifest})
    lines = [
        "# WildClaw Mixed-Category Evaluation",
        "",
        "## Scope",
        "",
        f"- Categories: {', '.join(f'`{item}`' for item in categories)}",
        f"- Manifest rows: {len(manifest)}",
        "- Conditions: `fixed_size_chunks`, `file_based_chunks`, `semantic_subcontext`",
        "",
        "## Category Summary",
        "",
        "| category | condition | tasks | missing workspaces | avg context chars | avg reduction |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            (
                f"| `{row['category']}` | `{row['condition']}` | {row['task_count']} | "
                f"{row['workspace_missing_count']} | {row['avg_context_chars']} | {row['avg_context_reduction']} |"
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Search Retrieval rows use the available local WildClawBench workspace files.",
            "- Productivity and Safety rows are prompt-only if their workspace directories are absent in the local checkout.",
            "- The same manifest shape can be passed to `scripts/run_wildclaw_framework_eval.py`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-contexts", type=Path, required=True)
    parser.add_argument("--semantic-jsonl", type=Path, required=True)
    parser.add_argument("--fixed-jsonl", type=Path, required=True)
    parser.add_argument("--file-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/wildclaw_mixed_category_eval"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_jsonl(args.raw_contexts)
    semantic_by_task = group_by_task(read_jsonl(args.semantic_jsonl))
    fixed_by_task = group_by_task(read_jsonl(args.fixed_jsonl))
    file_by_task = group_by_task(read_jsonl(args.file_jsonl))

    manifest = []
    for raw in raw_rows:
        task_id = raw["task"]["task_id"]
        manifest.append(
            manifest_row(raw, "fixed_size_chunks", fixed_by_task.get(task_id, []), concat_segments(fixed_by_task.get(task_id, [])))
        )
        manifest.append(
            manifest_row(raw, "file_based_chunks", file_by_task.get(task_id, []), concat_segments(file_by_task.get(task_id, [])))
        )
        manifest.append(
            manifest_row(raw, "semantic_subcontext", semantic_by_task.get(task_id, []), concat_segments(semantic_by_task.get(task_id, [])))
        )

    summary_rows = summarize(manifest)
    write_jsonl(args.output_dir / "wildclaw_mixed_category_eval_manifest.jsonl", manifest)
    (args.output_dir / "wildclaw_mixed_category_eval_manifest.json").write_text(
        json.dumps(
            {
                "created_at": now_iso(),
                "raw_contexts": args.raw_contexts.as_posix(),
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "wildclaw_mixed_category_eval_summary.csv", summary_rows)
    write_plan(args.output_dir / "wildclaw_mixed_category_eval_plan.md", manifest, summary_rows)

    print("WildClaw mixed-category evaluation prepared")
    print("===========================================")
    print(f"Output dir: {args.output_dir}")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Categories: {', '.join(sorted({row['category'] for row in manifest}))}")
    print(f"Missing-workspace rows: {sum(1 for row in manifest if row['workspace_missing'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
