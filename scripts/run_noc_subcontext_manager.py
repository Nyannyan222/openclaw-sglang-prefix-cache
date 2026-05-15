#!/usr/bin/env python3
"""Run the NOC semantic sub-context manager prototype on WildClaw JSONL data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noc_context_manager import RequestProfile, ReuseDecisionEngine, SubContextRegistry, SubContextSelector


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


def flatten_decision(row: dict[str, Any]) -> dict[str, Any]:
    similarity = row.pop("similarity")
    for key, value in similarity.items():
        row[f"similarity_{key}"] = value
    return row


def write_report(path: Path, summary: dict[str, Any], selected: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    decision_counts: dict[str, int] = {}
    relation_counts: dict[str, int] = {}
    for decision in decisions:
        decision_counts[decision["decision"]] = decision_counts.get(decision["decision"], 0) + 1
        relation_counts[decision["relation_type"]] = relation_counts.get(decision["relation_type"], 0) + 1

    lines = [
        "# NOC Semantic Sub-context Manager Smoke Run",
        "",
        "## Registry Summary",
        "",
        f"- Sub-contexts: {summary['sub_context_count']}",
        f"- Tasks: {summary['task_count']}",
        f"- Categories: {summary['category_count']}",
        f"- Duplicate content hashes: {summary['duplicate_content_hash_count']}",
        "",
        "## Relevant vs Reusable",
        "",
        "The selected context list answers the relevance question: which sub-contexts",
        "should accompany this request. The reuse relation table answers a separate",
        "question: whether two sub-contexts are safe to reuse as equivalent evidence.",
        "",
        "| relation type | meaning | reuse eligibility |",
        "| --- | --- | --- |",
        "| `exact_duplicate` | content is nearly identical after normalization | yes |",
        "| `near_duplicate` | different wording but potentially equivalent information | yes, after judge |",
        "| `same_answer_utility` | either context can support the same answer/evidence role | yes |",
        "| `partial_overlap` | some information overlaps, but evidence role may differ | no / maybe |",
        "| `broad_topic` | same topic/domain but different use | no |",
        "| `unrelated` | no meaningful relation | no |",
        "",
        "## Selected Contexts",
        "",
        "| rank | score | sub-context | task | title | chars |",
        "| ---: | ---: | --- | --- | --- | ---: |",
    ]
    for rank, row in enumerate(selected, start=1):
        context = row["sub_context"]
        lines.append(
            (
                f"| {rank} | {row['selection_score']} | `{context['id']}` | "
                f"`{context['task_id']}` | {context['title']} | {context['chars']} |"
            )
        )

    lines.extend(["", "## Reuse Relation Summary", ""])
    if relation_counts:
        lines.extend(["| relation type | count |", "| --- | ---: |"])
        for relation, count in sorted(relation_counts.items()):
            lines.append(f"| `{relation}` | {count} |")
    else:
        lines.append("No pair relations were generated.")

    lines.extend(["", "## Decision Summary", ""])
    if decision_counts:
        lines.extend(["| decision | count |", "| --- | ---: |"])
        for decision, count in sorted(decision_counts.items()):
            lines.append(f"| `{decision}` | {count} |")
    else:
        lines.append("No pair decisions were generated.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This prototype separates relevance from reusability. Embedding or lexical",
            "similarity can find candidates, but the reuse decision remains conservative.",
            "Only exact duplicates are immediately reusable. Near duplicates require",
            "embedding/LLM judge confirmation, and same-answer utility should be required",
            "before treating two sub-contexts as reusable evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/noc_subcontext_manager_smoke"))
    parser.add_argument("--request-id", default="smoke_request")
    parser.add_argument("--query", default="Find task-relevant evidence and avoid unsafe semantic reuse.")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--required-capability", action="append", default=[])
    parser.add_argument("--max-contexts", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--same-category-only", action="store_true")
    parser.add_argument("--different-task-only", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    registry = SubContextRegistry.from_wildclaw_jsonl(args.semantic_jsonl)
    request = RequestProfile(
        request_id=args.request_id,
        query=args.query,
        task_id=args.task_id,
        category=args.category,
        required_capabilities=args.required_capability,
        max_contexts=args.max_contexts,
        max_chars=args.max_chars,
    )
    selector = SubContextSelector(registry)
    selected = selector.select(request)
    engine = ReuseDecisionEngine(registry)
    decisions = [
        decision.to_dict()
        for decision in engine.decide_all_pairs(
            same_category_only=args.same_category_only,
            different_task_only=args.different_task_only,
            max_pairs=args.max_pairs,
        )
    ]
    flat_decisions = [flatten_decision(dict(decision)) for decision in decisions]
    summary = registry.summary()

    (args.output_dir / "subcontext_registry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "selected_subcontexts.jsonl", selected)
    write_jsonl(args.output_dir / "reuse_decisions.jsonl", decisions)
    write_csv(args.output_dir / "reuse_decisions.csv", flat_decisions)
    write_report(args.output_dir / "noc_subcontext_manager_report.md", summary, selected, decisions)

    print("NOC semantic sub-context manager")
    print("================================")
    print(f"Input: {args.semantic_jsonl}")
    print(f"Output dir: {args.output_dir}")
    print(f"Sub-contexts: {summary['sub_context_count']}")
    print(f"Selected: {len(selected)}")
    print(f"Reuse decisions: {len(decisions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
