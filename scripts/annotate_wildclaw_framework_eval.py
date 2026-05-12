"""Annotate a WildClaw framework evaluation run with lightweight heuristics.

This is a first-pass reviewer for the 04_Search_Retrieval pilot. It does not
replace manual review, but it produces a consistent CSV and condition summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


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


def read_text(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_web_sources(value: str) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def score_row(row: dict[str, str], answer: str) -> tuple[float, int, str]:
    task_id = row["task_id"]
    text = answer.lower()
    has_web = bool(row.get("web_source_urls", "").strip())

    if task_id.endswith("task_2_conflicting_handling"):
        has_three_years = bool(re.search(r"三年|3\s*年|three years", answer, re.I))
        mentions_old_two_years = bool(re.search(r"二年|2\s*年|two years|outdated|旧|previous", answer, re.I))
        correctness = 1.0 if has_three_years else 0.0
        evidence = 5 if correctness and has_web and mentions_old_two_years else 4 if correctness and has_web else 2
        note = "Correct current limitation period; stronger if it explains local two-year law is outdated."
        return correctness, evidence, note

    if task_id.endswith("task_4_efficient_search"):
        has_version = "3.12" in answer
        has_expected_gh = bool(re.search(r"\b90385\b|gh-90385|#90385", answer, re.I))
        has_wrong_pr = bool(re.search(r"\b119573\b", answer))
        if has_version and has_expected_gh:
            return 1.0, 5 if has_web else 4, "Has Python 3.12 and official gh-90385 evidence."
        if has_version:
            note = "Partial: has Python 3.12 but did not confirm the official gh-90385 reference."
            if has_wrong_pr:
                note += " It reports #119573, which appears to be unrelated follow-up work."
            return 0.5, 3 if has_web else 2, note
        return 0.0, 1, "Missing Python 3.12 and GitHub evidence."

    if task_id.endswith("task_5_fuzzy_search"):
        has_title = "visual-rft" in text or "visual reinforcement fine-tuning" in text
        has_author = "liu" in text
        has_stars = bool(re.search(r"2k|2,?300|over 2|more than 2|stars", text))
        if has_title and has_author and has_stars:
            return 1.0, 5 if has_web else 4, "Identifies Visual-RFT, Liu, and >2k GitHub stars."
        if has_title:
            return 0.5, 3 if has_web else 2, "Partial: identifies Visual-RFT but misses author or star evidence."
        return 0.0, 1, "Does not identify the target paper."

    return 0.0, 1, "No heuristic available for this task."


def numeric(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)

    summary = []
    for condition, items in sorted(groups.items()):
        summary.append(
            {
                "condition": condition,
                "rows": len(items),
                "avg_correctness": round(mean(float(item["answer_correctness_0_to_1"]) for item in items), 3),
                "avg_evidence_usefulness": round(mean(float(item["evidence_usefulness_1_to_5"]) for item in items), 3),
                "avg_context_reduction": round(mean(numeric(item["context_reduction"]) for item in items), 3),
                "avg_prompt_tokens": round(mean(numeric(item["prompt_tokens"]) for item in items), 1),
                "avg_completion_tokens": round(mean(numeric(item["completion_tokens"]) for item in items), 1),
                "avg_latency_s": round(mean(numeric(item["latency_s"]) for item in items), 3),
            }
        )
    return summary


def write_report(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_groups[row["task_id"]].append(row)

    lines = [
        "# WildClaw Retrieval/Search Execution Run Summary",
        "",
        "This run uses a two-stage `openai-web` backend: focused web retrieval first, final answer second.",
        "",
        "## Condition Summary",
        "",
        "| condition | rows | avg correctness | avg evidence | avg context reduction | avg prompt tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['condition']} | {item['rows']} | {item['avg_correctness']} | "
            f"{item['avg_evidence_usefulness']} | {item['avg_context_reduction']} | {item['avg_prompt_tokens']} |"
        )
    lines.extend(["", "## Task Notes", ""])
    for task_id, items in sorted(task_groups.items()):
        avg_score = mean(float(item["answer_correctness_0_to_1"]) for item in items)
        notes = sorted({item["notes"] for item in items})
        lines.append(f"- `{task_id}`: avg correctness {avg_score:.3f}. {' '.join(notes)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    results_path = args.run_dir / "wildclaw_framework_eval_results.csv"
    rows = read_csv(results_path)
    annotated: list[dict[str, Any]] = []
    for row in rows:
        answer = read_text(row.get("answer_path", ""))
        correctness, evidence, note = score_row(row, answer)
        sources = parse_web_sources(row.get("web_sources", ""))
        updated = dict(row)
        updated["answer_correctness_0_to_1"] = correctness
        updated["evidence_usefulness_1_to_5"] = evidence
        updated["web_source_count"] = len(sources)
        updated["notes"] = note
        annotated.append(updated)

    summary = summarize(annotated)
    write_csv(args.run_dir / "wildclaw_framework_eval_results_web_annotated.csv", annotated)
    write_csv(args.run_dir / "wildclaw_framework_eval_condition_summary_web_annotated.csv", summary)
    write_report(args.run_dir / "run_summary_web.md", annotated, summary)
    print(f"Annotated rows: {len(annotated)}")
    print(f"Wrote: {args.run_dir / 'run_summary_web.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
