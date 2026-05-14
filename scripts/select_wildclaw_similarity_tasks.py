#!/usr/bin/env python3
"""Rank WildClawBench tasks by semantic-similarity discovery potential.

The selector scans WildClawBench task markdown and local workspace files, then
scores tasks that are likely to contain repeated or near-repeated evidence. It
does not decide semantic equivalence itself; it finds good next candidates for
the stricter embedding + LLM judge pipeline.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from real_context_extractor import (
    parse_task_markdown,
    read_source_file,
    task_short_name,
    workspace_path_for,
)


DOMAIN_HINTS = {
    "api_docs": ("api", "sdk", "endpoint", "schema", "reference", "openapi"),
    "code_docs": ("readme", "docs", "documentation", "changelog", "release", "version"),
    "law_policy": ("law", "laws", "legal", "regulation", "policy", "compliance", "法", "条", "规定", "政策"),
    "safety": ("safety", "security", "prompt injection", "jailbreak", "leak", "安全"),
    "instructions": ("instruction", "requirement", "constraint", "rubric", "format", "guideline"),
    "tables": ("csv", "xlsx", "spreadsheet", "table"),
}


@dataclass(frozen=True)
class Segment:
    task_id: str
    source_path: str
    segment_id: str
    char_start: int
    char_end: int
    chars: int
    text: str
    normalized_hash: str
    shingles: frozenset[str]


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_shingles(text: str, n: int = 7, max_shingles: int = 1500) -> frozenset[str]:
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return frozenset()
    if len(compact) <= n:
        return frozenset({compact})
    shingles = {compact[index : index + n] for index in range(0, len(compact) - n + 1)}
    if len(shingles) <= max_shingles:
        return frozenset(shingles)
    # Deterministic downsample so large legal/source files do not dominate CPU.
    stride = max(1, math.ceil(len(shingles) / max_shingles))
    return frozenset(sorted(shingles)[::stride][:max_shingles])


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return overlap / len(left | right)


def split_long_piece(piece: str, base_start: int, max_chars: int, overlap_chars: int) -> list[tuple[int, int, str]]:
    windows = []
    start = 0
    while start < len(piece):
        end = min(len(piece), start + max_chars)
        chunk = piece[start:end].strip()
        if chunk:
            leading_ws = len(piece[start:end]) - len(piece[start:end].lstrip())
            windows.append((base_start + start + leading_ws, base_start + end, chunk))
        if end == len(piece):
            break
        start = max(start + 1, end - overlap_chars)
    return windows


def segment_source(
    task_id: str,
    source_path: str,
    text: str,
    min_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> list[Segment]:
    pieces = [part for part in re.split(r"\n\s*\n+", text.replace("\r\n", "\n").replace("\r", "\n")) if part.strip()]
    raw_segments: list[tuple[int, int, str]] = []
    cursor = 0
    for piece in pieces or [text]:
        start = text.find(piece, cursor)
        if start < 0:
            start = cursor
        end = start + len(piece)
        cursor = end
        if len(piece.strip()) > max_chars:
            raw_segments.extend(split_long_piece(piece, start, max_chars, overlap_chars))
        else:
            raw_segments.append((start, end, piece.strip()))

    segments = []
    for index, (start, end, segment_text) in enumerate(raw_segments, start=1):
        normalized = normalize_text(segment_text)
        if len(normalized) < min_chars:
            continue
        segments.append(
            Segment(
                task_id=task_id,
                source_path=source_path,
                segment_id=f"{source_path}::seg_{index:04d}",
                char_start=start,
                char_end=end,
                chars=len(segment_text),
                text=segment_text,
                normalized_hash=stable_hash(normalized),
                shingles=text_shingles(normalized),
            )
        )
    return segments


def discover_tasks(tasks_root: Path, categories: list[str]) -> list[Path]:
    if categories:
        task_files = []
        for category in categories:
            task_files.extend(sorted((tasks_root / category).glob("*.md")))
        return task_files
    return sorted(path for path in tasks_root.rglob("*.md") if path.name != "task0_template.md")


def domain_hints_for(task: dict[str, Any], sources: list[dict[str, Any]]) -> list[str]:
    haystack = "\n".join(
        [
            task.get("task_id", ""),
            task.get("name", ""),
            task.get("category", ""),
            task.get("prompt", "")[:5000],
            "\n".join(src.get("path", "") for src in sources),
            "\n".join((src.get("content") or "")[:1500] for src in sources if src.get("included_in_context")),
        ]
    ).lower()
    hints = []
    for hint, needles in DOMAIN_HINTS.items():
        if any(needle.lower() in haystack for needle in needles):
            hints.append(hint)
    return hints


def duplicate_groups(segments: list[Segment]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Segment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.normalized_hash].append(segment)
    rows = []
    for digest, items in grouped.items():
        source_paths = sorted({item.source_path for item in items})
        if len(items) < 2 or len(source_paths) < 2:
            continue
        rows.append(
            {
                "hash": digest,
                "count": len(items),
                "source_count": len(source_paths),
                "sources": source_paths[:6],
                "preview": items[0].text[:240].replace("\n", " "),
            }
        )
    return sorted(rows, key=lambda row: (row["source_count"], row["count"]), reverse=True)


def duplicate_file_groups(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        if source.get("included_in_context") and (source.get("content") or "").strip():
            grouped[source.get("sha256", "")].append(source)
    rows = []
    for digest, items in grouped.items():
        if digest and len(items) >= 2:
            rows.append(
                {
                    "sha256": digest,
                    "count": len(items),
                    "paths": [item.get("path", "") for item in items[:8]],
                }
            )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def near_duplicate_pairs(segments: list[Segment], threshold: float, max_segments: int, max_pairs: int) -> list[dict[str, Any]]:
    selected = sorted(segments, key=lambda item: item.chars, reverse=True)[:max_segments]
    pairs = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            if left.source_path == right.source_path:
                continue
            similarity = jaccard(left.shingles, right.shingles)
            if similarity < threshold:
                continue
            pairs.append(
                {
                    "left_source": left.source_path,
                    "right_source": right.source_path,
                    "left_segment": left.segment_id,
                    "right_segment": right.segment_id,
                    "similarity": round(similarity, 4),
                    "left_chars": left.chars,
                    "right_chars": right.chars,
                    "left_preview": left.text[:220].replace("\n", " "),
                    "right_preview": right.text[:220].replace("\n", " "),
                }
            )
    pairs.sort(key=lambda row: row["similarity"], reverse=True)
    return pairs[:max_pairs]


def prompt_terms(prompt: str) -> Counter[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{3,}|[\u4e00-\u9fff]{2,}", prompt.lower())
    stop = {
        "answer",
        "based",
        "complete",
        "following",
        "please",
        "question",
        "should",
        "task",
        "write",
    }
    return Counter(word for word in words if word not in stop)


def score_task(
    task: dict[str, Any],
    workspace_missing: bool,
    sources: list[dict[str, Any]],
    segments: list[Segment],
    exact_groups: list[dict[str, Any]],
    file_groups: list[dict[str, Any]],
    near_pairs: list[dict[str, Any]],
    hints: list[str],
) -> tuple[float, list[str]]:
    included_files = [src for src in sources if src.get("included_in_context") and (src.get("content") or "").strip()]
    total_chars = sum(len(src.get("content") or "") for src in included_files)
    score = 0.0
    reasons = []

    if workspace_missing:
        score -= 30.0
        reasons.append("workspace missing locally")
    else:
        score += 8.0
        reasons.append("workspace available")

    if included_files:
        score += min(len(included_files) * 1.5, 24.0)
        score += min(total_chars / 5000.0, 24.0)
        reasons.append(f"{len(included_files)} included workspace files")

    if file_groups:
        score += min(len(file_groups) * 12.0, 36.0)
        reasons.append(f"{len(file_groups)} exact duplicate file groups")

    if exact_groups:
        score += min(len(exact_groups) * 10.0, 40.0)
        reasons.append(f"{len(exact_groups)} exact duplicate segment groups")

    if near_pairs:
        top_similarity = near_pairs[0]["similarity"]
        score += min(len(near_pairs) * 5.0, 35.0)
        score += max(0.0, (top_similarity - 0.50) * 40.0)
        reasons.append(f"{len(near_pairs)} near-duplicate segment pairs; top={top_similarity:.3f}")

    high_value_hints = [hint for hint in hints if hint in {"api_docs", "code_docs", "law_policy", "safety", "instructions"}]
    if high_value_hints:
        score += len(high_value_hints) * 4.0
        reasons.append("domain hints: " + ", ".join(high_value_hints))

    prompt_overlap = sum(prompt_terms(task.get("prompt", "")).values())
    if prompt_overlap >= 80:
        score += 4.0
        reasons.append("long/constraint-heavy prompt")

    if not segments and not workspace_missing:
        score -= 25.0
        reasons.append("no usable text segments after filtering")

    return round(score, 3), reasons


def analyze_task(task_md: Path, args: argparse.Namespace) -> dict[str, Any]:
    task = parse_task_markdown(task_md)
    workspace_dir = workspace_path_for(args.workspace_root, task)
    workspace_missing = not workspace_dir.exists()

    source_files = []
    if not workspace_missing:
        for path in sorted(workspace_dir.rglob("*")):
            if path.is_file():
                source_files.append(asdict(read_source_file(path, workspace_dir, args.max_file_bytes, args.max_chars_per_file)))

    segments = []
    for source in source_files:
        if source.get("included_in_context") and source.get("content"):
            segments.extend(
                segment_source(
                    task["task_id"],
                    source["path"],
                    source["content"],
                    args.min_segment_chars,
                    args.max_segment_chars,
                    args.segment_overlap_chars,
                )
            )

    exact_groups = duplicate_groups(segments)
    file_groups = duplicate_file_groups(source_files)
    near_pairs = near_duplicate_pairs(segments, args.near_duplicate_threshold, args.max_segments_per_task, args.max_pairs_per_task)
    hints = domain_hints_for(task, source_files)
    score, reasons = score_task(task, workspace_missing, source_files, segments, exact_groups, file_groups, near_pairs, hints)
    included_files = [src for src in source_files if src.get("included_in_context") and (src.get("content") or "").strip()]
    skipped_files = [src for src in source_files if not src.get("included_in_context")]
    total_chars = sum(len(src.get("content") or "") for src in included_files)

    return {
        "task_id": task["task_id"],
        "name": task.get("name", ""),
        "category": task["category"],
        "task_markdown_path": task_md.as_posix(),
        "workspace_dir": workspace_dir.as_posix(),
        "workspace_missing": workspace_missing,
        "included_files": len(included_files),
        "skipped_files": len(skipped_files),
        "included_chars": total_chars,
        "segment_count": len(segments),
        "duplicate_file_groups": len(file_groups),
        "exact_duplicate_segment_groups": len(exact_groups),
        "near_duplicate_pairs": len(near_pairs),
        "max_near_duplicate_similarity": near_pairs[0]["similarity"] if near_pairs else 0.0,
        "domain_hints": hints,
        "selection_score": score,
        "selection_reasons": reasons,
        "top_duplicate_file_groups": file_groups[: args.max_details],
        "top_exact_duplicate_segment_groups": exact_groups[: args.max_details],
        "top_near_duplicate_pairs": near_pairs[: args.max_details],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "selection_score",
        "task_id",
        "category",
        "name",
        "workspace_missing",
        "included_files",
        "included_chars",
        "segment_count",
        "duplicate_file_groups",
        "exact_duplicate_segment_groups",
        "near_duplicate_pairs",
        "max_near_duplicate_similarity",
        "domain_hints",
        "selection_reasons",
        "workspace_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            flat = {key: row.get(key) for key in fieldnames}
            flat["rank"] = rank
            flat["domain_hints"] = "; ".join(row.get("domain_hints", []))
            flat["selection_reasons"] = "; ".join(row.get("selection_reasons", []))
            writer.writerow(flat)


def write_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    selected = [row for row in rows if row["selection_score"] >= args.min_select_score][: args.select_top]
    lines = [
        "# WildClaw Similarity Task Selector",
        "",
        f"- Created at: `{now_iso()}`",
        f"- WildClaw root: `{args.wildclaw_root.as_posix()}`",
        f"- Tasks scanned: {len(rows)}",
        f"- Selected top tasks: {len(selected)}",
        f"- Minimum selected score: {args.min_select_score}",
        "",
        "## Top Candidates",
        "",
        "| rank | score | task | category | files | segments | exact dup groups | near dup pairs | top sim | hints |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(selected, start=1):
        lines.append(
            (
                f"| {rank} | {row['selection_score']} | `{row['task_id']}` | `{row['category']}` | "
                f"{row['included_files']} | {row['segment_count']} | {row['exact_duplicate_segment_groups']} | "
                f"{row['near_duplicate_pairs']} | {row['max_near_duplicate_similarity']} | "
                f"{', '.join(row['domain_hints']) or '-'} |"
            )
        )

    lines.extend(
        [
            "",
            "## Why These Tasks",
            "",
        ]
    )
    for rank, row in enumerate(selected, start=1):
        lines.append(f"### {rank}. `{row['task_id']}`")
        lines.append("")
        lines.append(f"- Score: `{row['selection_score']}`")
        lines.append(f"- Reasons: {'; '.join(row['selection_reasons']) or 'n/a'}")
        if row["top_near_duplicate_pairs"]:
            top = row["top_near_duplicate_pairs"][0]
            lines.append(
                f"- Strongest near-duplicate pair: `{top['left_source']}` vs `{top['right_source']}` "
                f"(similarity `{top['similarity']}`)"
            )
        if row["top_exact_duplicate_segment_groups"]:
            group = row["top_exact_duplicate_segment_groups"][0]
            lines.append(f"- Exact duplicate segment sources: {', '.join(f'`{item}`' for item in group['sources'])}")
        lines.append("")

    task_args = " ".join(f"--task {row['task_id']}" for row in selected)
    lines.extend(
        [
            "## Suggested Next Commands",
            "",
            "Use the selected tasks as the next real-context extraction input:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\real_context_extractor.py `",
            "  --wildclaw-root external\\WildClawBench `",
            "  --output-dir benchmark_results\\wildclaw_similarity_selected_pilot `",
        ]
    )
    for row in selected:
        lines.append(f"  --task {row['task_id']} `")
    if selected:
        lines[-1] = lines[-1].rstrip(" `")
    lines.extend(
        [
            "```",
            "",
            "Then run semantic segmentation and strict similarity discovery on that selected pilot.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wildclaw-root", type=Path, default=Path("external/WildClawBench"))
    parser.add_argument("--tasks-root", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--category", action="append", default=[], help="Restrict to one category. Repeat for multiple.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/wildclaw_similarity_task_selection"))
    parser.add_argument("--select-top", type=int, default=8)
    parser.add_argument("--min-select-score", type=float, default=1.0)
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    parser.add_argument("--max-chars-per-file", type=int, default=80_000)
    parser.add_argument("--min-segment-chars", type=int, default=120)
    parser.add_argument("--max-segment-chars", type=int, default=2200)
    parser.add_argument("--segment-overlap-chars", type=int, default=250)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.42)
    parser.add_argument("--max-segments-per-task", type=int, default=260)
    parser.add_argument("--max-pairs-per-task", type=int, default=50)
    parser.add_argument("--max-details", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.wildclaw_root = args.wildclaw_root.resolve()
    args.tasks_root = (args.tasks_root.resolve() if args.tasks_root else args.wildclaw_root / "tasks")
    args.workspace_root = (args.workspace_root.resolve() if args.workspace_root else args.wildclaw_root / "workspace")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_files = discover_tasks(args.tasks_root, args.category)
    rows = [analyze_task(task_md, args) for task_md in task_files]
    rows.sort(
        key=lambda row: (
            row["selection_score"],
            row["near_duplicate_pairs"],
            row["exact_duplicate_segment_groups"],
            row["included_files"],
        ),
        reverse=True,
    )

    selected = [row for row in rows if row["selection_score"] >= args.min_select_score][: args.select_top]
    write_csv(args.output_dir / "wildclaw_similarity_task_candidates.csv", rows)
    (args.output_dir / "wildclaw_similarity_task_candidates.json").write_text(
        json.dumps({"created_at": now_iso(), "selector_args": vars(args), "tasks": rows}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (args.output_dir / "wildclaw_similarity_selected_tasks.txt").write_text(
        "\n".join(row["task_id"] for row in selected) + ("\n" if selected else ""),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "wildclaw_similarity_task_selection_report.md", rows, args)

    print("WildClaw similarity task selector")
    print("=================================")
    print(f"Tasks scanned: {len(rows)}")
    print(f"Output dir: {args.output_dir}")
    print("Top candidates:")
    for rank, row in enumerate(selected, start=1):
        print(
            f"{rank:>2}. {row['task_id']} "
            f"score={row['selection_score']} near_pairs={row['near_duplicate_pairs']} "
            f"exact_groups={row['exact_duplicate_segment_groups']} files={row['included_files']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
