#!/usr/bin/env python3
"""Summarize WildClaw SGLang runtime replay CSV results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


CONDITION_ORDER = [
    "fixed_size_chunks",
    "file_based_chunks",
    "candidate_evidence_context",
    "v3_semantic_final_pass",
    "canonical_context",
    "similar_context",
    "canonical_plus_delta",
    "canonical_context_repeat",
]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def condition_sort_key(condition: str) -> tuple[int, str]:
    try:
        return (CONDITION_ORDER.index(condition), condition)
    except ValueError:
        return (len(CONDITION_ORDER), condition)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def avg(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def summarize_condition_replay(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("condition", ""), row.get("replay_index", ""))].append(row)

    summaries = []
    for (condition, replay_index), group in sorted(
        groups.items(), key=lambda item: (condition_sort_key(item[0][0]), int(item[0][1] or 0))
    ):
        prompt_values = [parse_float(row.get("prompt_tokens")) for row in group]
        cached_values = [parse_float(row.get("cached_tokens")) for row in group]
        prefill_values = [parse_float(row.get("estimated_prefill_tokens")) for row in group]
        prompt_sum = sum(value or 0 for value in prompt_values)
        cached_sum = sum(value or 0 for value in cached_values)
        prefill_sum = sum(value or 0 for value in prefill_values)
        has_cached_values = any(value is not None for value in cached_values)
        has_prefill_values = any(value is not None for value in prefill_values)
        summaries.append(
            {
                "condition": condition,
                "replay_index": replay_index,
                "rows": len(group),
                "prompt_avg": avg(prompt_values),
                "cached_avg": avg(cached_values),
                "prefill_avg": avg(prefill_values),
                "latency_avg": avg([parse_float(row.get("latency_s")) for row in group]),
                "prompt_sum": prompt_sum,
                "cached_sum": cached_sum,
                "prefill_sum": prefill_sum,
                "cached_ratio": cached_sum / prompt_sum if prompt_sum and has_cached_values else None,
                "prefill_ratio": prefill_sum / prompt_sum if prompt_sum and has_prefill_values else None,
            }
        )
    return summaries


def summarize_reduction(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    replay1 = [
        row
        for row in rows
        if row.get("status") == "completed" and row.get("replay_index") == "1"
    ]
    task_condition: dict[tuple[str, str], float] = {}
    for idx, row in enumerate(replay1):
        task_id = row.get("task_id") or f"row_{idx}"
        prompt_tokens = parse_float(row.get("prompt_tokens"))
        if prompt_tokens is not None:
            task_condition[(task_id, row.get("condition", ""))] = prompt_tokens

    by_condition: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (task_id, condition), prompt_tokens in task_condition.items():
        fixed_tokens = task_condition.get((task_id, "fixed_size_chunks"))
        if fixed_tokens:
            by_condition[condition].append((fixed_tokens, prompt_tokens))

    summaries = []
    for condition, pairs in sorted(by_condition.items(), key=lambda item: condition_sort_key(item[0])):
        fixed_sum = sum(fixed for fixed, _ in pairs)
        condition_sum = sum(value for _, value in pairs)
        reductions = [(fixed - value) / fixed for fixed, value in pairs if fixed]
        summaries.append(
            {
                "condition": condition,
                "tasks": len(pairs),
                "fixed_prompt_sum": fixed_sum,
                "condition_prompt_sum": condition_sum,
                "weighted_reduction": (fixed_sum - condition_sum) / fixed_sum if fixed_sum else None,
                "mean_task_reduction": mean(reductions) if reductions else None,
            }
        )
    return summaries


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(csv_path: Path, rows: list[dict[str, str]]) -> str:
    completed = [row for row in rows if row.get("status") == "completed"]
    errors = [row for row in rows if row.get("status") != "completed"]
    condition_replay = summarize_condition_replay(completed)
    reductions = summarize_reduction(completed)

    lines = [
        "# WildClaw SGLang Runtime Summary",
        "",
        f"- Source CSV: `{csv_path.as_posix()}`",
        f"- Rows: {len(rows)}",
        f"- Completed: {len(completed)}",
        f"- Errors: {len(errors)}",
        "",
        "## Condition And Replay Summary",
        "",
        markdown_table(
            [
                "condition",
                "replay",
                "rows",
                "avg prompt",
                "avg cached",
                "avg prefill",
                "cached ratio",
                "avg latency s",
            ],
            [
                [
                    item["condition"],
                    str(item["replay_index"]),
                    str(item["rows"]),
                    fmt_num(item["prompt_avg"]),
                    fmt_num(item["cached_avg"]),
                    fmt_num(item["prefill_avg"]),
                    fmt_pct(item["cached_ratio"]),
                    fmt_num(item["latency_avg"], 4),
                ]
                for item in condition_replay
            ],
        ),
        "",
        "## Replay-1 Context Reduction Vs Fixed Size",
        "",
        markdown_table(
            [
                "condition",
                "tasks",
                "fixed prompt sum",
                "condition prompt sum",
                "weighted reduction",
                "mean task reduction",
            ],
            [
                [
                    item["condition"],
                    str(item["tasks"]),
                    fmt_num(item["fixed_prompt_sum"]),
                    fmt_num(item["condition_prompt_sum"]),
                    fmt_pct(item["weighted_reduction"]),
                    fmt_pct(item["mean_task_reduction"]),
                ]
                for item in reductions
            ],
        ),
        "",
        "## Notes",
        "",
        "- Replay 2 measures ordinary SGLang prefix-cache reuse on identical prompts.",
        "- The first real generation can include CUDA/Triton JIT warmup, so latency comparisons should be interpreted with warmup effects in mind.",
        "- `v3_semantic_final_pass` is expected to reduce prompt tokens when the semantic segmenter can isolate task-relevant sub-context.",
    ]

    if errors:
        lines.extend(
            [
                "",
                "## Errors",
                "",
                markdown_table(
                    ["task_id", "condition", "replay", "error"],
                    [
                        [
                            row.get("task_id", ""),
                            row.get("condition", ""),
                            row.get("replay_index", ""),
                            (row.get("error", "") or "")[:240].replace("\n", " "),
                        ]
                        for row in errors
                    ],
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to wildclaw_sglang_runtime_results.csv")
    parser.add_argument("--output", type=Path, default=None, help="Markdown output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.csv_path)
    output = args.output or args.csv_path.with_name("wildclaw_sglang_runtime_summary.md")
    output.write_text(render_markdown(args.csv_path, rows), encoding="utf-8")
    errors = sum(1 for row in rows if row.get("status") != "completed")
    print(f"Wrote {output}")
    print(f"Rows: {len(rows)}")
    print(f"Errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
