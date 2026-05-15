"""Registry for semantic sub-context lookup and grouping."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schema import SubContext


class SubContextRegistry:
    """In-memory registry of semantic sub-contexts.

    The registry is intentionally simple: it can be populated from existing
    WildClaw JSONL artifacts and queried by id, task, category, or content hash.
    A runtime implementation can later swap this for a persistent store.
    """

    def __init__(self, contexts: Iterable[SubContext] = ()) -> None:
        self._items: dict[str, SubContext] = {}
        self._by_hash: dict[str, list[str]] = defaultdict(list)
        self._by_task: dict[str, list[str]] = defaultdict(list)
        self._by_category: dict[str, list[str]] = defaultdict(list)
        for context in contexts:
            self.add(context)

    @classmethod
    def from_wildclaw_jsonl(cls, path: Path) -> "SubContextRegistry":
        contexts: list[SubContext] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    contexts.append(SubContext.from_wildclaw_row(json.loads(line)))
        return cls(contexts)

    def add(self, context: SubContext) -> None:
        if context.id in self._items:
            raise ValueError(f"duplicate sub-context id: {context.id}")
        self._items[context.id] = context
        self._by_hash[context.content_hash].append(context.id)
        if context.task_id:
            self._by_task[context.task_id].append(context.id)
        if context.category:
            self._by_category[context.category].append(context.id)

    def get(self, context_id: str) -> SubContext:
        return self._items[context_id]

    def all(self) -> list[SubContext]:
        return list(self._items.values())

    def by_task(self, task_id: str) -> list[SubContext]:
        return [self._items[item] for item in self._by_task.get(task_id, [])]

    def by_category(self, category: str) -> list[SubContext]:
        return [self._items[item] for item in self._by_category.get(category, [])]

    def exact_reuse_candidates(self, context: SubContext) -> list[SubContext]:
        return [
            self._items[item]
            for item in self._by_hash.get(context.content_hash, [])
            if item != context.id
        ]

    def summary(self) -> dict[str, object]:
        duplicate_hashes = {
            digest: ids
            for digest, ids in self._by_hash.items()
            if len(ids) > 1
        }
        return {
            "sub_context_count": len(self._items),
            "task_count": len(self._by_task),
            "category_count": len(self._by_category),
            "duplicate_content_hash_count": len(duplicate_hashes),
            "tasks": {task_id: len(ids) for task_id, ids in sorted(self._by_task.items())},
            "categories": {category: len(ids) for category, ids in sorted(self._by_category.items())},
        }
