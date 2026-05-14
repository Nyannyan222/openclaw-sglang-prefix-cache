#!/usr/bin/env python3
"""Summarize semantic similarity canonicalization runtime results per pair."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


CONDITION_ORDER = [
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


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


def pair_id(row: dict[str, str]) -> str:
    manifest_id = row.get("manifest_id", "")
    if "::" in manifest_id:
        return manifest_id.split("::", 1)[0]
    prompt_path = row.get("prompt_path", "")
    name = Path(prompt_path).name
    if "__" in name:
        return name.split("__", 1)[0]
    return row.get("task_id", "")


def avg(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def summarize_pairs(rows: list[dict[str, str]], pair_metadata: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in completed:
        groups[(pair_id(row), row.get("condition", ""))].append(row)

    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (pid, condition), group in groups.items():
        prompts = [parse_float(row.get("prompt_tokens")) for row in group]
        cached = [parse_float(row.get("cached_tokens")) for row in group]
        prefill = [parse_float(row.get("estimated_prefill_tokens")) for row in group]
        latency = [parse_float(row.get("latency_s")) for row in group]
        prompt_sum = sum(value or 0 for value in prompts)
        cached_sum = sum(value or 0 for value in cached)
        prefill_sum = sum(value or 0 for value in prefill)
        by_pair[pid][condition] = {
            "rows": len(group),
            "avg_prompt": avg(prompts),
            "avg_cached": avg(cached),
            "avg_prefill": avg(prefill),
            "avg_latency_s": avg(latency),
            "cached_ratio": cached_sum / prompt_sum if prompt_sum and any(value is not None for value in cached) else None,
            "prefill_ratio": prefill_sum / prompt_sum if prompt_sum and any(value is not None for value in prefill) else None,
        }

    summaries: list[dict[str, Any]] = []
    for pid in sorted(by_pair):
        conditions = by_pair[pid]
        similar = conditions.get("similar_context", {})
        proposed = conditions.get("canonical_plus_delta", {})
        repeat = conditions.get("canonical_context_repeat", {})
        canonical = conditions.get("canonical_context", {})
        metadata = pair_metadata.get(pid, {})
        similar_ratio = similar.get("cached_ratio")
        proposed_ratio = proposed.get("cached_ratio")
        similar_prefill = similar.get("avg_prefill")
        proposed_prefill = proposed.get("avg_prefill")
        similar_latency = similar.get("avg_latency_s")
        proposed_latency = proposed.get("avg_latency_s")
        summaries.append(
            {
                "pair_id": pid,
                "similarity": metadata.get("combined_similarity", ""),
                "delta_chars": metadata.get("delta_chars", ""),
                "canonical_prompt": canonical.get("avg_prompt"),
                "similar_prompt": similar.get("avg_prompt"),
                "canonical_plus_delta_prompt": proposed.get("avg_prompt"),
                "similar_cached_ratio": similar_ratio,
                "canonical_plus_delta_cached_ratio": proposed_ratio,
                "cached_ratio_gain": (
                    proposed_ratio - similar_ratio
                    if proposed_ratio is not None and similar_ratio is not None
                    else None
                ),
                "similar_prefill": similar_prefill,
                "canonical_plus_delta_prefill": proposed_prefill,
                "prefill_delta": (
                    proposed_prefill - similar_prefill
                    if proposed_prefill is not None and similar_prefill is not None
                    else None
                ),
                "similar_latency_s": similar_latency,
                "canonical_plus_delta_latency_s": proposed_latency,
                "latency_delta_s": (
                    proposed_latency - similar_latency
                    if proposed_latency is not None and similar_latency is not None
                    else None
                ),
                "repeat_cached_ratio": repeat.get("cached_ratio"),
                "proposed_beats_similar_ratio": (
                    proposed_ratio > similar_ratio
                    if proposed_ratio is not None and similar_ratio is not None
                    else None
                ),
            }
        )
    return summaries


def read_pair_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows = read_csv(path)
    return {row.get("pair_id", ""): row for row in rows}


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(csv_path: Path, summaries: list[dict[str, Any]]) -> str:
    wins = sum(1 for row in summaries if row.get("proposed_beats_similar_ratio") is True)
    measured = sum(1 for row in summaries if row.get("proposed_beats_similar_ratio") is not None)
    avg_gain = avg([row.get("cached_ratio_gain") for row in summaries])
    avg_latency_delta = avg([row.get("latency_delta_s") for row in summaries])
    return "\n".join(
        [
            "# Semantic Similarity Runtime Pair Summary",
            "",
            f"- Source CSV: `{csv_path.as_posix()}`",
            f"- Pairs: {len(summaries)}",
            f"- Proposed cached-ratio wins: {wins}/{measured}",
            f"- Mean cached-ratio gain: {fmt_pct(avg_gain)}",
            f"- Mean latency delta: {fmt_num(avg_latency_delta, 4)} s",
            "",
            markdown_table(
                [
                    "pair",
                    "similarity",
                    "delta chars",
                    "similar ratio",
                    "canonical+delta ratio",
                    "gain",
                    "similar prefill",
                    "canonical+delta prefill",
                    "latency delta s",
                    "repeat ratio",
                ],
                [
                    [
                        row["pair_id"],
                        str(row.get("similarity", "")),
                        str(row.get("delta_chars", "")),
                        fmt_pct(row.get("similar_cached_ratio")),
                        fmt_pct(row.get("canonical_plus_delta_cached_ratio")),
                        fmt_pct(row.get("cached_ratio_gain")),
                        fmt_num(row.get("similar_prefill")),
                        fmt_num(row.get("canonical_plus_delta_prefill")),
                        fmt_num(row.get("latency_delta_s"), 4),
                        fmt_pct(row.get("repeat_cached_ratio")),
                    ]
                    for row in summaries
                ],
            ),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--pairs-csv", type=Path, default=Path("benchmark_results/semantic_similarity_kv_reuse/semantic_similarity_pairs.csv"))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv(args.csv_path)
    summaries = summarize_pairs(rows, read_pair_metadata(args.pairs_csv))
    output_csv = args.output_csv or args.csv_path.with_name("semantic_similarity_runtime_pair_summary.csv")
    output_md = args.output_md or args.csv_path.with_name("semantic_similarity_runtime_pair_summary.md")
    write_csv(output_csv, summaries)
    output_md.write_text(render_markdown(args.csv_path, summaries), encoding="utf-8")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")
    print(f"Pairs: {len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
