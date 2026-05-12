#!/usr/bin/env python3
"""Prepare the next WildClawBench framework-evaluation experiment.

This script turns the reviewed V3 pilot into a clean final dataset and a
condition-by-condition manifest for evaluating a framework against:

1. fixed-size chunks,
2. file-based chunks,
3. candidate evidence contexts,
4. reviewed V3 semantic sub-contexts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def read_review(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return {row["id"]: row for row in csv.DictReader(fh)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
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


def group_candidate_contexts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["task"]["task_id"]: row for row in rows}


def task_prompt(raw_contexts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["task"]["task_id"]: row for row in raw_contexts}


def enrich_with_review(row: dict[str, Any], review: dict[str, str]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["manual_review_status"] = review.get("manual_review_status", "")
    enriched["manual_independence_1_to_5"] = review.get("manual_independence_1_to_5", "")
    enriched["manual_relevance_1_to_5"] = review.get("manual_relevance_1_to_5", "")
    enriched["manual_completeness_1_to_5"] = review.get("manual_completeness_1_to_5", "")
    enriched["manual_redundancy_1_to_5"] = review.get("manual_redundancy_1_to_5", "")
    enriched["review_notes"] = review.get("review_notes", "")
    return enriched


def concat_contexts(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        parts.append(
            "\n".join(
                [
                    f"[SUB_CONTEXT id={row.get('id')} title={row.get('title')}]",
                    row.get("sub_context", ""),
                    "[/SUB_CONTEXT]",
                ]
            )
        )
    return "\n\n".join(parts)


def make_eval_prompt(task: dict[str, Any], context_policy: str) -> str:
    return (
        "Use the provided context according to the context_policy. "
        "Do not use hidden grading answers. Complete the task objective and "
        "report the evidence you used.\n\n"
        f"context_policy: {context_policy}\n"
        f"task_id: {task['task_id']}\n"
        f"task_name: {task.get('name', '')}\n\n"
        f"Original task prompt:\n{task.get('prompt', '')}"
    )


def manifest_row(
    task_raw: dict[str, Any],
    condition: str,
    chunks: list[dict[str, Any]],
    context_text: str,
    notes: str,
) -> dict[str, Any]:
    task = task_raw["task"]
    full_chars = task_raw["long_context_chars"]
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
        "context_text": context_text,
        "eval_prompt": make_eval_prompt(task, condition),
        "expected_output": "Framework answer plus evidence trace. For this preparation step, no framework has been executed yet.",
        "notes": notes,
    }


def summarize_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    summary = []
    for condition, group in sorted(grouped.items()):
        summary.append(
            {
                "condition": condition,
                "task_count": len(group),
                "total_chunks": sum(row["chunk_count"] for row in group),
                "avg_chunks_per_task": round(sum(row["chunk_count"] for row in group) / len(group), 2),
                "avg_context_chars": round(sum(row["context_chars"] for row in group) / len(group), 1),
                "avg_context_reduction": round(sum(row["context_reduction"] for row in group) / len(group), 4),
            }
        )
    return summary


def review_summary(final_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not final_rows:
        return {}
    return {
        "row_count": len(final_rows),
        "avg_manual_independence": round(
            sum(float(row["manual_independence_1_to_5"]) for row in final_rows) / len(final_rows),
            3,
        ),
        "avg_manual_relevance": round(
            sum(float(row["manual_relevance_1_to_5"]) for row in final_rows) / len(final_rows),
            3,
        ),
        "avg_manual_completeness": round(
            sum(float(row["manual_completeness_1_to_5"]) for row in final_rows) / len(final_rows),
            3,
        ),
        "avg_manual_redundancy": round(
            sum(float(row["manual_redundancy_1_to_5"]) for row in final_rows) / len(final_rows),
            3,
        ),
    }


def write_plan(path: Path, summary_rows: list[dict[str, Any]], review: dict[str, Any]) -> None:
    lines = [
        "# WildClawBench Next-Round Framework Experiment",
        "",
        "## Goal",
        "",
        "Evaluate the framework with real WildClawBench contexts using four context policies:",
        "",
        "1. `fixed_size_chunks`",
        "2. `file_based_chunks`",
        "3. `candidate_evidence_context`",
        "4. `v3_semantic_final_pass`",
        "",
        "## Prepared Inputs",
        "",
        "| Condition | Task Count | Total Chunks | Avg Chunks/Task | Avg Context Chars | Avg Context Reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            (
                f"| `{row['condition']}` | {row['task_count']} | {row['total_chunks']} | "
                f"{row['avg_chunks_per_task']} | {row['avg_context_chars']} | "
                f"{row['avg_context_reduction']} |"
            )
        )

    lines.extend(
        [
            "",
            "## Final V3 Dataset Quality",
            "",
            f"- Rows kept: {review.get('row_count', 0)}",
            f"- Avg independence: {review.get('avg_manual_independence', 0)} / 5",
            f"- Avg relevance: {review.get('avg_manual_relevance', 0)} / 5",
            f"- Avg completeness: {review.get('avg_manual_completeness', 0)} / 5",
            f"- Avg redundancy: {review.get('avg_manual_redundancy', 0)} / 5",
            "",
            "## Next Execution Step",
            "",
            "Run the framework once per row in `wildclaw_framework_eval_manifest_v3.jsonl`.",
            "For each run, record:",
            "",
            "- answer correctness or manual usefulness score,",
            "- cited evidence IDs / source paths,",
            "- prompt/context tokens if available,",
            "- latency and cache metrics if connected to the SGLang benchmark path.",
            "",
            "Use `wildclaw_framework_eval_results_template.csv` as the run log.",
            "",
            "The expected comparison is whether `v3_semantic_final_pass` preserves evidence",
            "quality while reducing context relative to file-based and fixed-size policies.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def results_template_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in manifest:
        rows.append(
            {
                "manifest_id": row["id"],
                "task_id": row["task_id"],
                "condition": row["condition"],
                "chunk_count": row["chunk_count"],
                "context_chars": row["context_chars"],
                "context_reduction": row["context_reduction"],
                "framework_run_status": "pending",
                "answer_path": "",
                "answer_correctness_0_to_1": "",
                "evidence_usefulness_1_to_5": "",
                "cited_source_paths": "",
                "prompt_tokens": "",
                "completion_tokens": "",
                "latency_s": "",
                "cached_token_ratio": "",
                "notes": "",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v3-dir",
        type=Path,
        default=Path("benchmark_results/wildclaw_semantic_subcontext_pilot_v3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/wildclaw_next_experiment"),
    )
    parser.add_argument(
        "--include-revise",
        action="store_true",
        help="Also create pass+revise dataset. The main final dataset always keeps pass rows only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_contexts = read_jsonl(args.v3_dir / "wildclaw_real_contexts_pilot.jsonl")
    semantic_rows = read_jsonl(args.v3_dir / "wildclaw_semantic_subcontext_pilot.jsonl")
    fixed_rows = read_jsonl(args.v3_dir / "wildclaw_fixed_size_chunks_pilot.jsonl")
    file_rows = read_jsonl(args.v3_dir / "wildclaw_file_based_chunks_pilot.jsonl")
    candidate_contexts = read_jsonl(args.v3_dir / "wildclaw_candidate_evidence_contexts_v3.jsonl")
    review = read_review(args.v3_dir / "wildclaw_manual_review_sheet_v3_reviewed_by_codex.csv")

    enriched_semantic = [
        enrich_with_review(row, review[row["id"]])
        for row in semantic_rows
        if row["id"] in review
    ]
    pass_rows = [row for row in enriched_semantic if row["manual_review_status"] == "pass"]
    pass_revise_rows = [
        row
        for row in enriched_semantic
        if row["manual_review_status"] in {"pass", "revise"}
    ]

    final_path = args.output_dir / "wildclaw_semantic_subcontext_v3_final_pass.jsonl"
    final_revise_path = args.output_dir / "wildclaw_semantic_subcontext_v3_final_pass_plus_revise.jsonl"
    write_jsonl(final_path, pass_rows)
    write_jsonl(final_revise_path, pass_revise_rows)

    raw_by_task = task_prompt(raw_contexts)
    fixed_by_task = group_by_task(fixed_rows)
    file_by_task = group_by_task(file_rows)
    final_by_task = group_by_task(pass_rows)
    candidate_by_task = group_candidate_contexts(candidate_contexts)

    manifest = []
    for task_id, raw in raw_by_task.items():
        manifest.append(
            manifest_row(
                raw,
                "fixed_size_chunks",
                fixed_by_task.get(task_id, []),
                concat_contexts(fixed_by_task.get(task_id, [])),
                "Baseline: fixed-size character chunks.",
            )
        )
        manifest.append(
            manifest_row(
                raw,
                "file_based_chunks",
                file_by_task.get(task_id, []),
                concat_contexts(file_by_task.get(task_id, [])),
                "Baseline: prompt and each workspace file as chunks.",
            )
        )
        candidate = candidate_by_task.get(task_id)
        candidate_text = candidate["long_context"] if candidate else raw["task"].get("prompt", "")
        candidate_count = candidate.get("candidate_evidence_count", 0) if candidate else 0
        manifest.append(
            manifest_row(
                raw,
                "candidate_evidence_context",
                candidate.get("source_files", []) if candidate else [],
                candidate_text,
                f"Candidate evidence retrieval context. candidate_evidence_count={candidate_count}.",
            )
        )
        manifest.append(
            manifest_row(
                raw,
                "v3_semantic_final_pass",
                final_by_task.get(task_id, []),
                concat_contexts(final_by_task.get(task_id, [])),
                "Proposed condition: reviewed V3 pass-only semantic sub-contexts.",
            )
        )

    manifest_path = args.output_dir / "wildclaw_framework_eval_manifest_v3.jsonl"
    summary_path = args.output_dir / "wildclaw_next_experiment_condition_summary.csv"
    results_template_path = args.output_dir / "wildclaw_framework_eval_results_template.csv"
    review_json_path = args.output_dir / "wildclaw_v3_final_review_summary.json"
    plan_path = args.output_dir / "wildclaw_next_experiment_plan.md"
    manifest_json_path = args.output_dir / "wildclaw_framework_eval_manifest_v3.json"

    summary_rows = summarize_manifest(manifest)
    review_stats = review_summary(pass_rows)
    write_jsonl(manifest_path, manifest)
    write_csv(summary_path, summary_rows)
    write_csv(results_template_path, results_template_rows(manifest))
    review_json_path.write_text(json.dumps(review_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_json_path.write_text(
        json.dumps(
            {
                "created_at": now_iso(),
                "source_v3_dir": args.v3_dir.as_posix(),
                "final_dataset": final_path.as_posix(),
                "final_pass_plus_revise_dataset": final_revise_path.as_posix(),
                "summary": summary_rows,
                "review_summary": review_stats,
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_plan(plan_path, summary_rows, review_stats)

    print("WildClawBench next experiment prepared")
    print("=====================================")
    print(f"Final pass dataset: {final_path}")
    print(f"Pass+revise dataset: {final_revise_path}")
    print(f"Framework manifest: {manifest_path}")
    print(f"Condition summary: {summary_path}")
    print(f"Results template: {results_template_path}")
    print(f"Plan: {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
