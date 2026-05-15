"""Data structures for NOC semantic sub-context management."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text.strip()


def stable_content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SubContext:
    """A semantically independent context unit extracted from a real task."""

    id: str
    text: str
    content_hash: str
    source: str = ""
    method: str = ""
    category: str = ""
    task_id: str = ""
    task_name: str = ""
    title: str = ""
    question_or_objective: str = ""
    source_spans: list[dict[str, Any]] = field(default_factory=list)
    expected_capability: list[str] = field(default_factory=list)
    independence_score: float | None = None
    relevance_score: float | None = None
    completeness_score: float | None = None
    redundancy_risk: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wildclaw_row(cls, row: dict[str, Any]) -> "SubContext":
        text = row.get("sub_context") or row.get("text") or ""
        return cls(
            id=str(row.get("id") or row.get("sub_context_id") or stable_content_hash(text)[:16]),
            text=text,
            content_hash=str(row.get("content_hash") or stable_content_hash(text)),
            source=str(row.get("source") or ""),
            method=str(row.get("method") or ""),
            category=str(row.get("category") or ""),
            task_id=str(row.get("task_id") or ""),
            task_name=str(row.get("task_name") or ""),
            title=str(row.get("title") or ""),
            question_or_objective=str(row.get("question_or_objective") or ""),
            source_spans=list(row.get("source_spans") or []),
            expected_capability=list(row.get("expected_capability") or []),
            independence_score=as_optional_float(row.get("independence_score")),
            relevance_score=as_optional_float(row.get("relevance_score")),
            completeness_score=as_optional_float(row.get("completeness_score")),
            redundancy_risk=as_optional_float(row.get("redundancy_risk")),
            metadata={
                key: value
                for key, value in row.items()
                if key
                not in {
                    "id",
                    "sub_context_id",
                    "sub_context",
                    "text",
                    "content_hash",
                    "source",
                    "method",
                    "category",
                    "task_id",
                    "task_name",
                    "title",
                    "question_or_objective",
                    "source_spans",
                    "expected_capability",
                    "independence_score",
                    "relevance_score",
                    "completeness_score",
                    "redundancy_risk",
                }
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class RequestProfile:
    """Information available when selecting sub-contexts for an LLM request."""

    request_id: str
    query: str
    task_id: str = ""
    category: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    max_contexts: int = 6
    max_chars: int = 12_000


@dataclass(frozen=True)
class SimilaritySignal:
    left_id: str
    right_id: str
    lexical_score: float
    token_jaccard: float
    char_shingle_cosine: float
    exact_hash_match: bool
    same_task: bool
    same_category: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReuseDecision:
    """A conservative decision about whether one sub-context can reuse another."""

    left_id: str
    right_id: str
    decision: str
    reason: str
    safe_to_reuse: bool
    requires_manual_or_llm_review: bool
    similarity: SimilaritySignal

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["similarity"] = self.similarity.to_dict()
        return row


def as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
