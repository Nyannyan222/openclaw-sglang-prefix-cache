#!/usr/bin/env python3
"""Build candidate-evidence contexts from WildClawBench real contexts.

V3 narrows the LLM segmentation input from entire task workspaces to prompt plus
evidence-like passages. This is useful when the raw workspace is too large or
contains many unrelated legal/source documents.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DOMAIN_KEYWORDS = {
    "04_Search_Retrieval_task_2_conflicting_handling": [
        "statute of limitations",
        "limitation period",
        "claim period",
        "debt",
        "acknowledgment",
        "payment",
        "lawsuit",
        "诉讼时效",
        "时效期间",
        "请求权",
        "民事权利",
        "债务",
        "债权",
        "欠款",
        "履行期限",
        "中断",
        "中止",
        "重新计算",
        "三年",
        "二年",
    ],
    "04_Search_Retrieval_task_4_efficient_search": [
        "pathlib.Path.walk",
        "Python version",
        "standard library",
        "CPython",
        "pull request",
        "PR",
        "searches",
        "evidence chain",
        "Unable to confirm",
    ],
    "04_Search_Retrieval_task_5_fuzzy_search": [
        "DeepSeek-R1",
        "visual perception",
        "2025",
        "Liu",
        "GitHub",
        "2k stars",
        "paper",
        "repository",
        "Visual-RFT",
    ],
}

REQUIRED_KEYWORDS = {
    "04_Search_Retrieval_task_2_conflicting_handling": [
        "诉讼时效",
        "时效期间",
        "请求权",
        "民事权利",
    ],
}


@dataclass(frozen=True)
class EvidencePassage:
    task_id: str
    source_path: str
    passage_id: str
    score: float
    matched_keywords: list[str]
    char_start: int
    char_end: int
    text: str


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


def prompt_keywords(prompt: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{3,}", prompt)
    stop = {
        "after",
        "answer",
        "based",
        "business",
        "complete",
        "determine",
        "following",
        "please",
        "question",
        "results",
        "should",
        "within",
        "write",
    }
    seen: set[str] = set()
    keywords = []
    for word in words:
        lower = word.lower()
        if lower in stop or lower in seen:
            continue
        seen.add(lower)
        keywords.append(word)
    return keywords[:30]


def keyword_set(task: dict[str, Any]) -> list[str]:
    task_id = task["task_id"]
    combined = [*DOMAIN_KEYWORDS.get(task_id, []), *prompt_keywords(task.get("prompt", ""))]
    deduped = []
    seen: set[str] = set()
    for keyword in combined:
        key = keyword.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(keyword)
    return deduped


def passage_windows(text: str, max_chars: int, overlap_chars: int) -> list[tuple[int, int, str]]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    windows: list[tuple[int, int, str]] = []
    if len(paragraphs) >= 2:
        cursor = 0
        for paragraph in paragraphs:
            start = text.find(paragraph, cursor)
            if start < 0:
                start = cursor
            end = start + len(paragraph)
            cursor = end
            if len(paragraph) <= max_chars:
                windows.append((start, end, paragraph))
                continue
            windows.extend(sliding_windows(paragraph, start, max_chars, overlap_chars))
        return windows
    return sliding_windows(text, 0, max_chars, overlap_chars)


def sliding_windows(text: str, base_start: int, max_chars: int, overlap_chars: int) -> list[tuple[int, int, str]]:
    windows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            windows.append((base_start + start, base_start + end, chunk))
        if end == len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return windows


def score_passage(text: str, keywords: list[str]) -> tuple[float, list[str]]:
    lowered = text.lower()
    matches = []
    score = 0.0
    for keyword in keywords:
        key = keyword.lower()
        if key in lowered:
            matches.append(keyword)
            score += 2.0 if len(keyword) >= 6 else 1.0
    if re.search(r"(第[一二三四五六七八九十百千\d]+条|Article\s+\d+)", text):
        score += 1.5
    if re.search(r"(三年|二年|3\s*years?|2\s*years?)", text, flags=re.IGNORECASE):
        score += 1.5
    return score, matches


def has_required_keyword(task_id: str, text: str) -> bool:
    required = REQUIRED_KEYWORDS.get(task_id)
    if not required:
        return True
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in required)


def collect_evidence(raw: dict[str, Any], args: argparse.Namespace) -> list[EvidencePassage]:
    task = raw["task"]
    task_id = task["task_id"]
    keywords = keyword_set(task)
    candidates: list[EvidencePassage] = []
    for source in raw.get("source_files", []):
        text = source.get("content") or ""
        if not source.get("included_in_context") or not text.strip():
            continue
        for index, (start, end, passage) in enumerate(passage_windows(text, args.max_passage_chars, args.overlap_chars), start=1):
            if not has_required_keyword(task_id, passage):
                continue
            score, matches = score_passage(passage, keywords)
            if score < args.min_score:
                continue
            candidates.append(
                EvidencePassage(
                    task_id=task_id,
                    source_path=source["path"],
                    passage_id=f"{source['path']}#p{index}",
                    score=score,
                    matched_keywords=matches[:20],
                    char_start=start,
                    char_end=end,
                    text=passage,
                )
            )
    candidates.sort(key=lambda item: (item.score, len(item.matched_keywords), -len(item.text)), reverse=True)
    return candidates[: args.max_passages_per_task]


def build_candidate_long_context(raw: dict[str, Any], passages: list[EvidencePassage]) -> str:
    task = raw["task"]
    parts = [
        f"[TASK_ID]\n{task['task_id']}",
        f"[CATEGORY]\n{task['category']}",
        f"[TASK_NAME]\n{task.get('name') or ''}",
        f"[PROMPT]\n{task.get('prompt') or ''}",
        "[CANDIDATE_EVIDENCE_NOTE]\n"
        "The following passages were selected from real workspace files by keyword/evidence matching. "
        "Use them as candidate evidence for semantic sub-context extraction.",
    ]
    for item in passages:
        parts.append(
            "\n".join(
                [
                    (
                        f"[CANDIDATE_EVIDENCE path={item.source_path} score={item.score:.2f} "
                        f"char_start={item.char_start} char_end={item.char_end}]"
                    ),
                    f"matched_keywords={', '.join(item.matched_keywords)}",
                    item.text,
                    "[/CANDIDATE_EVIDENCE]",
                ]
            )
        )
    return "\n\n".join(parts)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_contexts_jsonl", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/wildclaw_semantic_subcontext_pilot_v3"),
    )
    parser.add_argument("--max-passages-per-task", type=int, default=18)
    parser.add_argument("--max-passage-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument("--min-score", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_jsonl(args.raw_contexts_jsonl)
    candidate_rows = []
    evidence_rows = []
    summary_rows = []

    for raw in raw_rows:
        passages = collect_evidence(raw, args)
        task = raw["task"]
        long_context = build_candidate_long_context(raw, passages)
        candidate_rows.append(
            {
                "schema_version": "wildclaw-candidate-evidence-v1",
                "source": "WildClawBench",
                "created_at": now_iso(),
                "task": task,
                "workspace_dir": raw.get("workspace_dir"),
                "source_files": [
                    {
                        "path": item.source_path,
                        "role": "candidate_evidence",
                        "score": item.score,
                        "matched_keywords": item.matched_keywords,
                        "char_start": item.char_start,
                        "char_end": item.char_end,
                        "included_in_context": True,
                        "content": item.text,
                    }
                    for item in passages
                ],
                "long_context": long_context,
                "long_context_chars": len(long_context),
                "candidate_evidence_count": len(passages),
            }
        )
        for item in passages:
            evidence_rows.append(asdict(item))
        summary_rows.append(
            {
                "task_id": task["task_id"],
                "candidate_evidence_count": len(passages),
                "candidate_context_chars": len(long_context),
                "top_sources": " | ".join(item.source_path for item in passages[:5]),
            }
        )

    contexts_path = args.output_dir / "wildclaw_candidate_evidence_contexts_v3.jsonl"
    evidence_path = args.output_dir / "wildclaw_candidate_evidence_passages_v3.jsonl"
    summary_path = args.output_dir / "wildclaw_candidate_evidence_summary_v3.csv"
    write_jsonl(contexts_path, candidate_rows)
    write_jsonl(evidence_path, evidence_rows)
    write_csv(summary_path, summary_rows)

    print("WildClawBench candidate evidence extraction")
    print("==========================================")
    print(f"Candidate contexts: {contexts_path}")
    print(f"Evidence passages: {evidence_path}")
    print(f"Summary: {summary_path}")
    for row in summary_rows:
        print(f"- {row['task_id']}: {row['candidate_evidence_count']} passages, {row['candidate_context_chars']} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
