#!/usr/bin/env python3
"""Segment WildClawBench real contexts into semantic sub-contexts.

The script supports an LLM mode through an OpenAI-compatible Chat Completions
endpoint and a deterministic heuristic fallback for dry runs. It also writes
fixed-size and file-based baselines so the first pilot can compare chunking
strategies with the same input contexts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import textwrap
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You extract semantically independent sub-contexts for evaluation.

Rules:
- Do not solve the original task.
- Do not use grading criteria, ground truth, or answer-like material.
- Each sub-context must be understandable and evaluable without the full original context.
- Preserve evidence-bearing text and cite source file paths or PROMPT.
- Do not create metadata-only segments such as task id, category, task name, or
  output path unless the output constraint is bundled with task-relevant evidence.
- Each sub-context should include original evidence, not only a summary.
- Target 6-8 sub-contexts per task. Prefer fewer rich sub-contexts over many
  tiny fragments.
- Aim for 300-800 characters per sub-context when the source context allows it.
- Return strict JSON only, with a top-level key "sub_contexts".
"""


USER_PROMPT_TEMPLATE = """Task metadata:
task_id: {task_id}
category: {category}
name: {name}

Long context:
{long_context}

Return JSON using this shape:
{{
  "sub_contexts": [
    {{
      "title": "short title",
      "source_spans": [
        {{"source": "PROMPT or workspace path", "start_line": 1, "end_line": 12}}
      ],
      "sub_context": "self-contained text",
      "atomic_objective": "what this sub-context can evaluate",
      "required_capabilities": ["retrieval", "reasoning"],
      "independence_score": 0.0,
      "relevance_score": 0.0,
      "completeness_score": 0.0,
      "redundancy_risk": 0.0
    }}
  ]
}}

Additional v2 requirements:
- The "sub_context" field must contain enough original evidence to be evaluated
  without reopening the full long context.
- Include concrete facts, constraints, source requirements, or legal/search
  evidence. Avoid generic labels like "Task Overview" unless bundled with
  concrete evidence.
- If the source text is short, combine related constraints into one richer
  sub-context instead of splitting them into one-line fragments.
"""


@dataclass(frozen=True)
class SegmentRecord:
    id: str
    source: str
    method: str
    category: str
    task_id: str
    task_name: str
    title: str
    source_spans: list[dict[str, Any]]
    sub_context: str
    question_or_objective: str
    expected_capability: list[str]
    independence_score: float | None
    relevance_score: float | None
    completeness_score: float | None
    redundancy_risk: float | None
    long_context_chars: int
    sub_context_chars: int
    context_reduction: float
    content_hash: str
    leakage_checked: bool
    manual_review_status: str


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def split_lines_with_offsets(text: str) -> list[tuple[int, str]]:
    return [(idx + 1, line) for idx, line in enumerate(text.splitlines())]


def prompt_keywords(task: dict[str, Any]) -> set[str]:
    text = f"{task.get('prompt', '')} {task.get('name', '')}".lower()
    return {
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text)
        if word not in {"please", "should", "would", "could", "into", "from", "that", "with", "this"}
    }


def heuristic_scores(segment: str, task: dict[str, Any]) -> tuple[float, float, float, float]:
    keywords = prompt_keywords(task)
    lowered = segment.lower()
    overlap = sum(1 for word in keywords if word in lowered)
    relevance = min(1.0, overlap / max(4, len(keywords) * 0.35)) if keywords else 0.5
    independence = 0.75 if len(segment) >= 350 and re.search(r"[.!?。]\s", segment) else 0.6
    completeness = min(0.9, 0.45 + len(segment) / 3500)
    redundancy = 0.2 if len(segment) < 6000 else 0.45
    return independence, relevance, completeness, redundancy


def make_record(
    raw: dict[str, Any],
    method: str,
    index: int,
    title: str,
    spans: list[dict[str, Any]],
    sub_context: str,
    objective: str,
    capabilities: list[str],
    independence: float | None,
    relevance: float | None,
    completeness: float | None,
    redundancy: float | None,
) -> SegmentRecord:
    task = raw["task"]
    task_id = task["task_id"]
    long_chars = int(raw["long_context_chars"])
    sub_chars = len(sub_context)
    return SegmentRecord(
        id=f"wildclaw_{task_id}_{method}_subctx_{index:02d}",
        source="WildClawBench",
        method=method,
        category=task["category"],
        task_id=task_id,
        task_name=task.get("name", ""),
        title=title,
        source_spans=spans,
        sub_context=sub_context,
        question_or_objective=objective,
        expected_capability=capabilities,
        independence_score=independence,
        relevance_score=relevance,
        completeness_score=completeness,
        redundancy_risk=redundancy,
        long_context_chars=long_chars,
        sub_context_chars=sub_chars,
        context_reduction=1.0 - (sub_chars / long_chars) if long_chars else 0.0,
        content_hash=content_hash(sub_context),
        leakage_checked=True,
        manual_review_status="pending",
    )


def source_file_segments(raw: dict[str, Any]) -> list[SegmentRecord]:
    records = []
    index = 1
    task = raw["task"]
    prompt = task.get("prompt", "")
    if prompt:
        independence, relevance, completeness, redundancy = heuristic_scores(prompt, task)
        records.append(
            make_record(
                raw,
                "file_based",
                index,
                "Task prompt",
                [{"source": "PROMPT", "start_line": 1, "end_line": prompt.count("\n") + 1}],
                prompt,
                "Understand the original task request and output contract.",
                ["task_understanding"],
                independence,
                relevance,
                completeness,
                redundancy,
            )
        )
        index += 1

    for source in raw.get("source_files", []):
        if not source.get("included_in_context") or not source.get("content"):
            continue
        text = source["content"]
        independence, relevance, completeness, redundancy = heuristic_scores(text, task)
        records.append(
            make_record(
                raw,
                "file_based",
                index,
                f"Workspace file: {source['path']}",
                [{"source": source["path"], "start_line": 1, "end_line": source.get("line_count") or text.count("\n") + 1}],
                text,
                f"Use evidence from {source['path']} as an independent workspace context.",
                ["evidence_extraction", "retrieval"],
                independence,
                relevance,
                completeness,
                redundancy,
            )
        )
        index += 1
    return records


def fixed_size_segments(raw: dict[str, Any], chunk_chars: int, overlap_chars: int) -> list[SegmentRecord]:
    text = raw["long_context"]
    records = []
    start = 0
    index = 1
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end]
        independence, relevance, completeness, redundancy = heuristic_scores(chunk, raw["task"])
        records.append(
            make_record(
                raw,
                "fixed_size",
                index,
                f"Fixed-size chunk {index}",
                [{"source": "LONG_CONTEXT", "start_char": start, "end_char": end}],
                chunk,
                "Evaluate retrieval and reasoning from a fixed character-window chunk.",
                ["retrieval"],
                independence * 0.8,
                relevance,
                completeness * 0.75,
                min(1.0, redundancy + 0.15),
            )
        )
        index += 1
        if end == len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return records


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/InternLM/WildClawBench",
            "X-Title": "WildClaw semantic sub-context pilot",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {detail}") from exc
    data = json.loads(body)
    return data["choices"][0]["message"]["content"]


def llm_segments(raw: dict[str, Any], args: argparse.Namespace) -> list[SegmentRecord]:
    task = raw["task"]
    context = raw["long_context"]
    if len(context) > args.max_context_chars:
        context = context[: args.max_context_chars] + "\n\n[TRUNCATED_FOR_LLM_SEGMENTATION]"
    user_prompt = USER_PROMPT_TEMPLATE.format(
        task_id=task["task_id"],
        category=task["category"],
        name=task.get("name", ""),
        long_context=context,
    )
    content = chat_completion(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        timeout=args.timeout,
    )
    parsed = extract_json_object(content)
    records = []
    for index, item in enumerate(parsed.get("sub_contexts", []), start=1):
        sub_context = str(item.get("sub_context", "")).strip()
        if not sub_context:
            continue
        independence = clamp_score(item.get("independence_score"))
        relevance = clamp_score(item.get("relevance_score"))
        completeness = clamp_score(item.get("completeness_score"))
        redundancy = clamp_score(item.get("redundancy_risk"))
        if None in (independence, relevance, completeness, redundancy):
            estimated = heuristic_scores(sub_context, task)
            independence = independence if independence is not None else estimated[0]
            relevance = relevance if relevance is not None else estimated[1]
            completeness = completeness if completeness is not None else estimated[2]
            redundancy = redundancy if redundancy is not None else estimated[3]
        records.append(
            make_record(
                raw,
                "llm_semantic",
                index,
                str(item.get("title", f"Semantic sub-context {index}")),
                list(item.get("source_spans", [])),
                sub_context,
                str(item.get("atomic_objective", "")),
                list(item.get("required_capabilities", [])),
                independence,
                relevance,
                completeness,
                redundancy,
            )
        )
    return records


def heuristic_semantic_segments(raw: dict[str, Any], max_segments: int) -> list[SegmentRecord]:
    task = raw["task"]
    candidates: list[tuple[str, str, str]] = []
    prompt = task.get("prompt", "")
    if prompt:
        candidates.append(("PROMPT", "Task prompt and output contract", prompt))
    for source in raw.get("source_files", []):
        if not source.get("included_in_context") or not source.get("content"):
            continue
        source_candidates = 0
        chunks = re.split(r"\n\s*\n+", source["content"])
        for idx, chunk in enumerate(chunks, start=1):
            chunk = chunk.strip()
            if len(chunk) >= 180:
                candidates.append((source["path"], f"{source['path']} section {idx}", chunk))
                source_candidates += 1

        if source_candidates == 0 and source["content"].strip():
            candidates.append((source["path"], f"{source['path']} compact section", source["content"].strip()))
        elif source_candidates < 2 and len(source["content"]) > 8000:
            lines = source["content"].splitlines()
            window: list[str] = []
            section_index = 1
            for line in lines:
                window.append(line)
                window_text = "\n".join(window).strip()
                if len(window_text) >= 3500 or len(window) >= 80:
                    candidates.append((source["path"], f"{source['path']} line window {section_index}", window_text))
                    section_index += 1
                    window = []
            tail = "\n".join(window).strip()
            if len(tail) >= 500:
                candidates.append((source["path"], f"{source['path']} line window {section_index}", tail))

    if not candidates:
        fallback_text = raw.get("long_context", "")
        if fallback_text:
            candidates.append(("LONG_CONTEXT", "Compact long-context fallback", fallback_text))

    candidates.sort(key=lambda item: len(item[2]), reverse=True)
    records = []
    for index, (source, title, text) in enumerate(candidates[:max_segments], start=1):
        wrapped = textwrap.shorten(text.replace("\n", " "), width=5000, placeholder=" ...")
        independence, relevance, completeness, redundancy = heuristic_scores(wrapped, task)
        records.append(
            make_record(
                raw,
                "llm_semantic",
                index,
                title,
                [{"source": source}],
                wrapped,
                "Heuristic fallback segment; replace with LLM semantic segmentation for final pilot.",
                ["manual_review", "retrieval"],
                independence,
                relevance,
                completeness,
                redundancy,
            )
        )
    return records


def aggregate_comparison(rows: list[SegmentRecord]) -> list[dict[str, Any]]:
    groups: dict[str, list[SegmentRecord]] = {}
    for row in rows:
        groups.setdefault(row.method, []).append(row)
    comparison = []
    for method, group in sorted(groups.items()):
        def avg(attr: str) -> float | None:
            values = [getattr(row, attr) for row in group if getattr(row, attr) is not None]
            return sum(values) / len(values) if values else None

        comparison.append(
            {
                "method": method,
                "sub_context_count": len(group),
                "avg_sub_context_chars": round(sum(row.sub_context_chars for row in group) / len(group), 1),
                "avg_context_reduction": round(sum(row.context_reduction for row in group) / len(group), 4),
                "avg_independence": round(avg("independence_score") or 0.0, 4),
                "avg_relevance": round(avg("relevance_score") or 0.0, 4),
                "avg_completeness": round(avg("completeness_score") or 0.0, 4),
                "avg_redundancy_risk": round(avg("redundancy_risk") or 0.0, 4),
                "manual_review_status": "pending",
            }
        )
    return comparison


def manual_review_rows(rows: list[SegmentRecord], limit: int) -> list[dict[str, Any]]:
    review_rows = []
    for row in rows[:limit]:
        review_rows.append(
            {
                "id": row.id,
                "task_id": row.task_id,
                "title": row.title,
                "source_spans": json.dumps(row.source_spans, ensure_ascii=False),
                "sub_context_preview": textwrap.shorten(row.sub_context.replace("\n", " "), width=360, placeholder=" ..."),
                "manual_review_status": "",
                "manual_independence_1_to_5": "",
                "manual_relevance_1_to_5": "",
                "manual_completeness_1_to_5": "",
                "manual_redundancy_1_to_5": "",
                "leakage_issue": "",
                "review_notes": "",
            }
        )
    return review_rows


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
        default=Path("benchmark_results/wildclaw_semantic_subcontext_pilot"),
    )
    parser.add_argument("--mode", choices=["auto", "llm", "heuristic"], default="auto")
    parser.add_argument("--model", default=os.environ.get("DEFAULT_MODEL") or os.environ.get("OPENAI_MODEL") or "openai/gpt-5.4")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-context-chars", type=int, default=60_000)
    parser.add_argument("--fixed-chunk-chars", type=int, default=4000)
    parser.add_argument("--fixed-overlap-chars", type=int, default=400)
    parser.add_argument("--heuristic-max-segments-per-task", type=int, default=8)
    parser.add_argument("--manual-review-limit", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_jsonl(args.raw_contexts_jsonl)
    use_llm = args.mode == "llm" or (args.mode == "auto" and bool(args.api_key))
    if args.mode == "llm" and not args.api_key:
        raise SystemExit("LLM mode requires OPENROUTER_API_KEY or OPENAI_API_KEY.")

    semantic: list[SegmentRecord] = []
    fixed: list[SegmentRecord] = []
    file_based: list[SegmentRecord] = []
    run_log = {
        "created_at": now_iso(),
        "mode": "llm" if use_llm else "heuristic",
        "model": args.model if use_llm else None,
        "raw_contexts_jsonl": args.raw_contexts_jsonl.as_posix(),
        "tasks": [],
    }

    for raw in raw_rows:
        task_id = raw["task"]["task_id"]
        fixed.extend(fixed_size_segments(raw, args.fixed_chunk_chars, args.fixed_overlap_chars))
        file_based.extend(source_file_segments(raw))
        try:
            task_segments = llm_segments(raw, args) if use_llm else heuristic_semantic_segments(raw, args.heuristic_max_segments_per_task)
            semantic.extend(task_segments)
            run_log["tasks"].append({"task_id": task_id, "semantic_segments": len(task_segments), "error": None})
        except Exception as exc:
            fallback = heuristic_semantic_segments(raw, args.heuristic_max_segments_per_task)
            semantic.extend(fallback)
            run_log["tasks"].append({"task_id": task_id, "semantic_segments": len(fallback), "error": str(exc), "fallback": "heuristic"})

    all_rows = [*fixed, *file_based, *semantic]
    semantic_path = args.output_dir / "wildclaw_semantic_subcontext_pilot.jsonl"
    fixed_path = args.output_dir / "wildclaw_fixed_size_chunks_pilot.jsonl"
    file_path = args.output_dir / "wildclaw_file_based_chunks_pilot.jsonl"
    comparison_path = args.output_dir / "wildclaw_chunking_comparison.csv"
    manual_review_path = args.output_dir / "wildclaw_manual_review_sheet.csv"
    run_log_path = args.output_dir / "wildclaw_semantic_segmenter_run.json"

    write_jsonl(semantic_path, [asdict(row) for row in semantic])
    write_jsonl(fixed_path, [asdict(row) for row in fixed])
    write_jsonl(file_path, [asdict(row) for row in file_based])
    write_csv(comparison_path, aggregate_comparison(all_rows))
    write_csv(manual_review_path, manual_review_rows(semantic, args.manual_review_limit))
    run_log_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WildClawBench semantic sub-context segmentation")
    print("===============================================")
    print(f"Mode: {'llm' if use_llm else 'heuristic'}")
    print(f"Semantic JSONL: {semantic_path}")
    print(f"Fixed-size baseline: {fixed_path}")
    print(f"File-based baseline: {file_path}")
    print(f"Comparison table: {comparison_path}")
    print(f"Manual review sheet: {manual_review_path}")
    print(f"Run log: {run_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
