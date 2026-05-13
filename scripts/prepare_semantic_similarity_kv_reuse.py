#!/usr/bin/env python3
"""Prepare a semantic/content-similarity KV reuse opportunity protocol.

Important: KV tensors are exact-token runtime state. This script does not claim
that semantically similar but token-different text can be safely spliced into
SGLang's KV cache. It prepares the measurable opportunity:

1. find similar sub-contexts by normalized content similarity,
2. build prompts that compare canonical text, similar text, canonical+delta,
   and canonical repeat,
3. produce a manifest for native SGLang replay,
4. document which pairs would require canonicalization or delta preservation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


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


def word_tokens(text: str) -> list[str]:
    lowered = text.lower()
    return re.findall(r"[\w\u4e00-\u9fff]{2,}", lowered)


def char_shingles(text: str, n: int = 5) -> Counter[str]:
    normalized = normalize_text(text)
    if not normalized:
        return Counter()
    if len(normalized) <= n:
        return Counter([normalized])
    return Counter(normalized[index : index + n] for index in range(0, len(normalized) - n + 1))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def similarity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_text = left.get("sub_context", "")
    right_text = right.get("sub_context", "")
    shingle_cosine = cosine(char_shingles(left_text), char_shingles(right_text))
    token_jaccard = jaccard(set(word_tokens(left_text)), set(word_tokens(right_text)))
    combined = 0.75 * shingle_cosine + 0.25 * token_jaccard
    return {
        "char_shingle_cosine": shingle_cosine,
        "token_jaccard": token_jaccard,
        "combined_similarity": combined,
    }


def split_units(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    rough_units = re.split(r"(?<=[.!?。！？；;])\s+|\n+", normalized)
    units = [unit.strip() for unit in rough_units if unit.strip()]
    if len(units) <= 1 and len(normalized) > 240:
        units = [normalized[index : index + 220].strip() for index in range(0, len(normalized), 220)]
    return [unit for unit in units if unit]


def unit_similarity(left: str, right: str) -> float:
    return similarity({"sub_context": left}, {"sub_context": right})["combined_similarity"]


def extract_delta(canonical_text: str, similar_text: str, unit_threshold: float = 0.72) -> str:
    canonical_units = split_units(canonical_text)
    similar_units = split_units(similar_text)
    delta_units = []
    for unit in similar_units:
        best = max((unit_similarity(unit, canon) for canon in canonical_units), default=0.0)
        if best < unit_threshold:
            delta_units.append(unit)
    if delta_units:
        return "\n".join(delta_units)
    if normalize_text(canonical_text) != normalize_text(similar_text):
        return similar_text
    return "No task-specific delta beyond the canonical context."


def pair_rows(rows: list[dict[str, Any]], threshold: float, same_category_only: bool) -> list[dict[str, Any]]:
    candidates = []
    usable = [row for row in rows if row.get("sub_context")]
    for i, left in enumerate(usable):
        for right in usable[i + 1 :]:
            if left.get("id") == right.get("id"):
                continue
            if same_category_only and left.get("category") != right.get("category"):
                continue
            if left.get("task_id") == right.get("task_id") and left.get("content_hash") == right.get("content_hash"):
                continue
            scores = similarity(left, right)
            if scores["combined_similarity"] < threshold:
                continue
            left_text = left.get("sub_context", "")
            right_text = right.get("sub_context", "")
            exact_token_safe = content_hash(left_text) == content_hash(right_text)
            normalized_exact = normalized_hash(left_text) == normalized_hash(right_text)
            candidates.append(
                {
                    "pair_id": f"sim_pair_{len(candidates) + 1:03d}",
                    "left_id": left.get("id", ""),
                    "right_id": right.get("id", ""),
                    "left_task_id": left.get("task_id", ""),
                    "right_task_id": right.get("task_id", ""),
                    "left_category": left.get("category", ""),
                    "right_category": right.get("category", ""),
                    "left_chars": len(left_text),
                    "right_chars": len(right_text),
                    "left_content_hash": content_hash(left_text),
                    "right_content_hash": content_hash(right_text),
                    "left_normalized_hash": normalized_hash(left_text),
                    "right_normalized_hash": normalized_hash(right_text),
                    "exact_token_safe_reuse": exact_token_safe,
                    "normalized_exact_match": normalized_exact,
                    "reuse_policy": (
                        "safe_exact_kv_reuse"
                        if exact_token_safe
                        else "canonicalize_before_kv_reuse"
                        if normalized_exact
                        else "semantic_similarity_candidate_no_direct_kv_reuse"
                    ),
                    **scores,
                    "left_text": left_text,
                    "right_text": right_text,
                }
            )
    candidates.sort(key=lambda row: row["combined_similarity"], reverse=True)
    return candidates


def prompt_for(pair: dict[str, Any], variant: str, text: str) -> str:
    # Put the evaluated text at the very beginning. This minimizes shared
    # instruction-prefix cache hits so cached_tokens mostly reflect the context
    # block itself.
    return "\n\n".join(
        [
            "[CONTEXT_BLOCK]",
            text,
            "[/CONTEXT_BLOCK]",
            (
                "Task: Based only on the context block above, give a one-sentence "
                "summary and say whether it is sufficient evidence. "
                f"Metadata: pair={pair['pair_id']}; variant={variant}; "
                f"similarity={pair['combined_similarity']:.4f}; "
                f"reuse_policy={pair['reuse_policy']}."
            ),
        ]
    )


def canonical_plus_delta_prompt(pair: dict[str, Any]) -> str:
    canonical_text = pair["left_text"]
    similar_text = pair["right_text"]
    delta_text = pair["delta_text"]
    return "\n\n".join(
        [
            "[CANONICAL_CONTEXT]",
            canonical_text,
            "[/CANONICAL_CONTEXT]",
            "[TASK_SPECIFIC_DELTA]",
            delta_text,
            "[/TASK_SPECIFIC_DELTA]",
            (
                "Task: Use the canonical context as reusable evidence and apply "
                "the task-specific delta to preserve details from the similar "
                "context. Give a one-sentence summary and say whether the combined "
                "context is sufficient evidence. "
                f"Metadata: pair={pair['pair_id']}; variant=canonical_plus_delta; "
                f"similarity={pair['combined_similarity']:.4f}; "
                f"reuse_policy=semantic_canonicalization_with_delta."
            ),
        ]
    )


def build_runtime(rows: list[dict[str, Any]], output_dir: Path, max_pairs: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    runtime_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for pair in rows[:max_pairs]:
        pair["delta_text"] = extract_delta(pair["left_text"], pair["right_text"])
        pair["delta_chars"] = len(pair["delta_text"])
        pair["delta_hash"] = content_hash(pair["delta_text"])
        variants = {
            "canonical_context": pair["left_text"],
            "similar_context": pair["right_text"],
            "canonical_plus_delta": None,
            "canonical_context_repeat": pair["left_text"],
        }
        for variant, text in variants.items():
            prompt = canonical_plus_delta_prompt(pair) if variant == "canonical_plus_delta" else prompt_for(pair, variant, text or "")
            prompt_path = prompt_dir / f"{pair['pair_id']}__{variant}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            kv_reuse_mode = (
                "semantic_canonicalization_with_delta"
                if variant == "canonical_plus_delta"
                else pair["reuse_policy"]
            )
            runtime_rows.append(
                {
                    "id": f"{pair['pair_id']}::{variant}",
                    "task_id": f"{pair['left_task_id']}__vs__{pair['right_task_id']}",
                    "condition": variant,
                    "prompt_path": prompt_path.as_posix(),
                    "kv_reuse_mode": kv_reuse_mode,
                    "similarity": round(pair["combined_similarity"], 6),
                    "notes": (
                        "canonical_plus_delta is the proposed method: reuse the exact canonical "
                        "prefix and preserve task-specific differences as a delta."
                        if variant == "canonical_plus_delta"
                        else "similar_context should not be treated as safe direct KV reuse unless "
                        "the reuse_policy is safe_exact_kv_reuse or a canonicalization layer "
                        "makes the runtime text/token ids identical."
                    ),
                }
            )
            block_text = pair["left_text"] if variant == "canonical_plus_delta" else text or ""
            block_rows.append(
                {
                    "runtime_row_id": f"{pair['pair_id']}::{variant}",
                    "pair_id": pair["pair_id"],
                    "variant": variant,
                    "content_hash": content_hash(block_text),
                    "normalized_hash": normalized_hash(block_text),
                    "chars": len(block_text),
                    "delta_chars": len(pair["delta_text"]) if variant == "canonical_plus_delta" else 0,
                    "reuse_policy": kv_reuse_mode,
                    "combined_similarity": round(pair["combined_similarity"], 6),
                }
            )
    return runtime_rows, block_rows


def strip_text_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped = []
    for row in rows:
        clean = dict(row)
        clean.pop("left_text", None)
        clean.pop("right_text", None)
        clean.pop("delta_text", None)
        stripped.append(clean)
    return stripped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/semantic_similarity_kv_reuse"))
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--same-category-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.semantic_jsonl)
    pairs = pair_rows(rows, threshold=args.threshold, same_category_only=args.same_category_only)
    runtime_rows, block_rows = build_runtime(pairs, args.output_dir, args.max_pairs)

    write_csv(args.output_dir / "semantic_similarity_pairs.csv", strip_text_fields(pairs[: args.max_pairs]))
    write_jsonl(args.output_dir / "semantic_similarity_runtime_manifest.jsonl", runtime_rows)
    write_jsonl(args.output_dir / "semantic_similarity_block_manifest.jsonl", block_rows)
    (args.output_dir / "semantic_similarity_protocol.json").write_text(
        json.dumps(
            {
                "created_at": now_iso(),
                "semantic_jsonl": args.semantic_jsonl.as_posix(),
                "threshold": args.threshold,
                "same_category_only": args.same_category_only,
                "runtime_manifest": (args.output_dir / "semantic_similarity_runtime_manifest.jsonl").as_posix(),
                "block_manifest": (args.output_dir / "semantic_similarity_block_manifest.jsonl").as_posix(),
                "runtime_principle": {
                    "direct_kv_reuse_requires": "same model, tokenizer, exact token ids, positions, and attention history",
                    "semantic_similarity_use": "candidate detection, semantic grouping, canonical context substitution, and task-specific delta preservation before runtime",
                    "proposed_condition": "canonical_plus_delta",
                    "unsafe_case": "do not splice KV for token-different paraphrases without a validated approximate-attention method",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Semantic/content similarity KV reuse protocol prepared")
    print("======================================================")
    print(f"Output dir: {args.output_dir}")
    print(f"Pairs selected: {min(len(pairs), args.max_pairs)} / {len(pairs)}")
    print(f"Runtime rows: {len(runtime_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
