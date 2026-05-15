"""Similarity utilities for conservative sub-context reuse decisions."""

from __future__ import annotations

import math
import re
from collections import Counter

from .schema import SimilaritySignal, SubContext, normalize_text


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


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class LexicalSimilarityJudge:
    """Offline similarity judge used before expensive embedding/LLM checks.

    This class does not claim semantic equivalence. It provides a cheap signal
    for ranking candidates and for exact-hash reuse decisions.
    """

    def score_texts(self, left: str, right: str) -> dict[str, float]:
        shingle_cosine = counter_cosine(char_shingles(left), char_shingles(right))
        token_jaccard = jaccard(set(word_tokens(left)), set(word_tokens(right)))
        lexical_score = 0.75 * shingle_cosine + 0.25 * token_jaccard
        return {
            "char_shingle_cosine": shingle_cosine,
            "token_jaccard": token_jaccard,
            "lexical_score": lexical_score,
        }

    def compare(self, left: SubContext, right: SubContext) -> SimilaritySignal:
        scores = self.score_texts(left.text, right.text)
        return SimilaritySignal(
            left_id=left.id,
            right_id=right.id,
            lexical_score=scores["lexical_score"],
            token_jaccard=scores["token_jaccard"],
            char_shingle_cosine=scores["char_shingle_cosine"],
            exact_hash_match=left.content_hash == right.content_hash,
            same_task=bool(left.task_id and left.task_id == right.task_id),
            same_category=bool(left.category and left.category == right.category),
        )
