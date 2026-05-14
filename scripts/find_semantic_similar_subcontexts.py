#!/usr/bin/env python3
"""Find genuinely semantically similar WildClaw sub-contexts.

This script replaces the canonical-plus-delta direction for the current
research stage. Its output is a semantic-similarity dataset: pairs and groups
of sub-contexts that are likely to express the same or highly overlapping
meaning/evidence role.

Lexical overlap is used only as a cheap prefilter. When an OpenAI API key is
available, embeddings and an optional LLM judge provide the semantic signal.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
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


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text.strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]{2,}", text.lower())


def char_shingles(text: str, n: int = 5) -> Counter[str]:
    normalized = normalize_text(text)
    if not normalized:
        return Counter()
    if len(normalized) <= n:
        return Counter([normalized])
    return Counter(normalized[index : index + n] for index in range(0, len(normalized) - n + 1))


def counter_cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def vector_cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def lexical_scores(left: str, right: str) -> dict[str, float]:
    shingle_cosine = counter_cosine(char_shingles(left), char_shingles(right))
    token_jaccard = jaccard(set(word_tokens(left)), set(word_tokens(right)))
    lexical_score = 0.75 * shingle_cosine + 0.25 * token_jaccard
    return {
        "char_shingle_cosine": shingle_cosine,
        "token_jaccard": token_jaccard,
        "lexical_score": lexical_score,
    }


def openai_json_request(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def embed_texts(texts: list[str], api_key: str, model: str, timeout: float, batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    endpoint = "https://api.openai.com/v1/embeddings"
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        data = openai_json_request(
            endpoint,
            api_key,
            {"model": model, "input": batch},
            timeout,
        )
        vectors.extend(item["embedding"] for item in sorted(data["data"], key=lambda item: item["index"]))
    return vectors


def judge_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    api_key: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    system = (
        "You judge whether two benchmark sub-contexts are semantically similar. "
        "Return strict JSON only. Focus on meaning, evidence role, and task relevance. "
        "Do not reward identical formatting alone. Do not mark two contexts as highly "
        "similar just because they belong to the same task. If they cover different "
        "parts of the task, label them as partial_overlap unless they can support "
        "substantially the same answer or evidence role."
    )
    user = {
        "left": {
            "title": left.get("title", ""),
            "task": left.get("task_id", ""),
            "objective": left.get("question_or_objective", ""),
            "sub_context": left.get("sub_context", ""),
        },
        "right": {
            "title": right.get("title", ""),
            "task": right.get("task_id", ""),
            "objective": right.get("question_or_objective", ""),
            "sub_context": right.get("sub_context", ""),
        },
        "rubric": {
            "score_0": "unrelated or contradictory",
            "score_1": "same broad topic only",
            "score_2": "same task/topic but different evidence role or different required detail",
            "score_3": "highly overlapping meaning/evidence role with only minor detail differences",
            "score_4": "near-equivalent meaning/evidence; either context could usually replace the other",
        },
        "required_json_schema": {
            "score": "integer 0-4",
            "relation": "unrelated | broad_topic | partial_overlap | high_similarity | near_equivalent",
            "same_answer_utility": "boolean",
            "rationale": "short string",
        },
    }
    data = openai_json_request(
        "https://api.openai.com/v1/chat/completions",
        api_key,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout,
    )
    content = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"score": None, "relation": "parse_error", "same_answer_utility": False, "rationale": content[:500]}
    return parsed


def choose_backend(requested: str, api_key: str | None) -> str:
    if requested != "auto":
        return requested
    return "openai_embedding" if api_key else "lexical_prefilter"


def pair_candidates(
    rows: list[dict[str, Any]],
    prefilter_threshold: float,
    same_category_only: bool,
    different_task_only: bool,
) -> list[dict[str, Any]]:
    usable = [row for row in rows if row.get("sub_context")]
    candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(usable):
        for right_index, right in enumerate(usable[left_index + 1 :], start=left_index + 1):
            if same_category_only and left.get("category") != right.get("category"):
                continue
            if different_task_only and left.get("task_id") == right.get("task_id"):
                continue
            left_text = left.get("sub_context", "")
            right_text = right.get("sub_context", "")
            scores = lexical_scores(left_text, right_text)
            if scores["lexical_score"] < prefilter_threshold:
                continue
            candidates.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_id": left.get("id", ""),
                    "right_id": right.get("id", ""),
                    "left_task_id": left.get("task_id", ""),
                    "right_task_id": right.get("task_id", ""),
                    "left_category": left.get("category", ""),
                    "right_category": right.get("category", ""),
                    "left_title": left.get("title", ""),
                    "right_title": right.get("title", ""),
                    "left_chars": len(left_text),
                    "right_chars": len(right_text),
                    "left_content_hash": content_hash(left_text),
                    "right_content_hash": content_hash(right_text),
                    **scores,
                }
            )
    candidates.sort(key=lambda row: row["lexical_score"], reverse=True)
    return candidates


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def semantic_decision(
    row: dict[str, Any],
    embedding_threshold: float,
    judge_threshold: int,
    min_match_embedding: float,
    require_same_answer_utility: bool,
) -> str:
    judge_score = row.get("llm_judge_score")
    embedding_score = row.get("embedding_cosine")
    if judge_score not in ("", None):
        judge_match = int(judge_score) >= judge_threshold
        utility_match = truthy(row.get("same_answer_utility"))
        embedding_match = (
            embedding_score not in ("", None)
            and float(embedding_score) >= min_match_embedding
        )
        if judge_match and (not require_same_answer_utility or utility_match) and embedding_match:
            return "semantic_match"
        if judge_match:
            return "semantic_related_needs_review"
        return "semantic_reject"
    if embedding_score not in ("", None):
        return "semantic_match" if float(embedding_score) >= embedding_threshold else "semantic_reject"
    return "needs_semantic_judge"


def assign_pair_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, row in enumerate(rows, start=1):
        row["pair_id"] = f"semantic_pair_{index:03d}"
    return rows


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def build_groups(pairs: list[dict[str, Any]], rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    uf = UnionFind()
    for pair in pairs:
        if pair.get("semantic_decision") == "semantic_match":
            uf.union(pair["left_id"], pair["right_id"])
    groups: dict[str, list[str]] = defaultdict(list)
    for item in sorted(uf.parent):
        groups[uf.find(item)].append(item)
    output = []
    for index, members in enumerate((members for members in groups.values() if len(members) > 1), start=1):
        titles = [rows_by_id[member].get("title", "") for member in members if member in rows_by_id]
        tasks = sorted({rows_by_id[member].get("task_id", "") for member in members if member in rows_by_id})
        output.append(
            {
                "group_id": f"semantic_group_{index:03d}",
                "member_count": len(members),
                "members": members,
                "tasks": tasks,
                "titles": titles,
            }
        )
    return output


def strip_text(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"left_text", "right_text"}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/semantic_similarity_discovery"))
    parser.add_argument("--backend", choices=["auto", "lexical_prefilter", "openai_embedding", "openai_embedding_judge"], default="auto")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--prefilter-threshold", type=float, default=0.02)
    parser.add_argument("--embedding-threshold", type=float, default=0.72)
    parser.add_argument("--judge-threshold", type=int, default=3)
    parser.add_argument("--min-match-embedding", type=float, default=0.50, help="Minimum embedding cosine required for an LLM-judged semantic_match.")
    parser.add_argument("--require-same-answer-utility", action="store_true", help="Require the LLM judge to say either context can support substantially the same answer/evidence role.")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--same-category-only", action="store_true")
    parser.add_argument("--different-task-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--sleep-between-judge", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    backend = choose_backend(args.backend, api_key)
    if backend.startswith("openai") and not api_key:
        raise SystemExit(f"{args.api_key_env} is required for backend={backend}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.semantic_jsonl)
    rows_by_id = {row.get("id", ""): row for row in rows}
    candidates = pair_candidates(
        rows,
        prefilter_threshold=args.prefilter_threshold,
        same_category_only=args.same_category_only,
        different_task_only=args.different_task_only,
    )[: args.max_candidates]

    embeddings: list[list[float]] = []
    if backend in {"openai_embedding", "openai_embedding_judge"}:
        texts = [row.get("sub_context", "") for row in rows]
        embeddings = embed_texts(texts, api_key or "", args.embedding_model, args.timeout, args.embedding_batch_size)

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        left = rows[candidate["left_index"]]
        right = rows[candidate["right_index"]]
        if embeddings:
            candidate["embedding_model"] = args.embedding_model
            candidate["embedding_cosine"] = vector_cosine(embeddings[candidate["left_index"]], embeddings[candidate["right_index"]])
        else:
            candidate["embedding_model"] = ""
            candidate["embedding_cosine"] = ""

        if backend == "openai_embedding_judge":
            judge = judge_pair(left, right, api_key or "", args.judge_model, args.timeout)
            candidate["llm_judge_model"] = args.judge_model
            candidate["llm_judge_score"] = judge.get("score", "")
            candidate["semantic_relation"] = judge.get("relation", "")
            candidate["same_answer_utility"] = judge.get("same_answer_utility", "")
            candidate["llm_judge_rationale"] = judge.get("rationale", "")
            if args.sleep_between_judge:
                time.sleep(args.sleep_between_judge)
        else:
            candidate["llm_judge_model"] = ""
            candidate["llm_judge_score"] = ""
            candidate["semantic_relation"] = ""
            candidate["same_answer_utility"] = ""
            candidate["llm_judge_rationale"] = ""

        candidate["semantic_decision"] = semantic_decision(
            candidate,
            args.embedding_threshold,
            args.judge_threshold,
            args.min_match_embedding,
            args.require_same_answer_utility,
        )
        candidate["left_text_preview"] = left.get("sub_context", "")[:300].replace("\n", " ")
        candidate["right_text_preview"] = right.get("sub_context", "")[:300].replace("\n", " ")
        scored.append(strip_text(candidate))

    scored.sort(
        key=lambda row: (
            row.get("semantic_decision") != "semantic_match",
            -(float(row["embedding_cosine"]) if row.get("embedding_cosine") not in ("", None) else float(row["lexical_score"])),
        )
    )
    assign_pair_ids(scored)
    groups = build_groups(scored, rows_by_id)

    write_jsonl(args.output_dir / "semantic_similar_subcontext_pairs.jsonl", scored)
    write_csv(args.output_dir / "semantic_similar_subcontext_pairs.csv", scored)
    write_jsonl(args.output_dir / "semantic_similar_subcontext_groups.jsonl", groups)
    write_csv(
        args.output_dir / "semantic_similarity_manual_review.csv",
        [
            {
                "pair_id": row["pair_id"],
                "semantic_decision": row["semantic_decision"],
                "embedding_cosine": row.get("embedding_cosine", ""),
                "llm_judge_score": row.get("llm_judge_score", ""),
                "semantic_relation": row.get("semantic_relation", ""),
                "manual_label": "",
                "manual_notes": "",
                "left_id": row["left_id"],
                "right_id": row["right_id"],
                "left_text_preview": row["left_text_preview"],
                "right_text_preview": row["right_text_preview"],
            }
            for row in scored
        ],
    )
    (args.output_dir / "semantic_similarity_discovery_protocol.json").write_text(
        json.dumps(
            {
                "created_at": now_iso(),
                "semantic_jsonl": args.semantic_jsonl.as_posix(),
                "backend": backend,
                "prefilter_threshold": args.prefilter_threshold,
                "embedding_model": args.embedding_model if backend.startswith("openai") else "",
                "embedding_threshold": args.embedding_threshold,
                "judge_model": args.judge_model if backend == "openai_embedding_judge" else "",
                "judge_threshold": args.judge_threshold if backend == "openai_embedding_judge" else "",
                "min_match_embedding": args.min_match_embedding if backend == "openai_embedding_judge" else "",
                "require_same_answer_utility": args.require_same_answer_utility if backend == "openai_embedding_judge" else "",
                "outputs": {
                    "pairs_csv": (args.output_dir / "semantic_similar_subcontext_pairs.csv").as_posix(),
                    "pairs_jsonl": (args.output_dir / "semantic_similar_subcontext_pairs.jsonl").as_posix(),
                    "groups_jsonl": (args.output_dir / "semantic_similar_subcontext_groups.jsonl").as_posix(),
                    "manual_review_csv": (args.output_dir / "semantic_similarity_manual_review.csv").as_posix(),
                },
                "note": "This pipeline finds semantic similarity only. It does not build canonical_plus_delta prompts or claim KV reuse.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    matches = sum(1 for row in scored if row["semantic_decision"] == "semantic_match")
    print("Semantic similar sub-context discovery complete")
    print("==============================================")
    print(f"Backend: {backend}")
    print(f"Candidates: {len(scored)}")
    print(f"Semantic matches: {matches}")
    print(f"Groups: {len(groups)}")
    print(f"Output dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
