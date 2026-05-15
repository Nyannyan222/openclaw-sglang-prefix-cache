"""Relevance selection for request-specific sub-context windows."""

from __future__ import annotations

from .registry import SubContextRegistry
from .schema import RequestProfile, SubContext
from .similarity import LexicalSimilarityJudge


class SubContextSelector:
    """Select a concise set of relevant sub-contexts for a request."""

    def __init__(self, registry: SubContextRegistry, judge: LexicalSimilarityJudge | None = None) -> None:
        self.registry = registry
        self.judge = judge or LexicalSimilarityJudge()

    def select(self, request: RequestProfile) -> list[dict[str, object]]:
        candidates = self._candidate_pool(request)
        scored = []
        used_chars = 0
        for context in candidates:
            score, reasons = self._score_context(request, context)
            if score <= 0:
                continue
            scored.append(
                {
                    "sub_context": context,
                    "selection_score": round(score, 4),
                    "selection_reasons": reasons,
                }
            )
        scored.sort(key=lambda row: row["selection_score"], reverse=True)

        selected = []
        for row in scored:
            context = row["sub_context"]
            if len(selected) >= request.max_contexts:
                break
            if used_chars + context.chars > request.max_chars and selected:
                continue
            used_chars += context.chars
            context_row = context.to_dict()
            context_row["chars"] = context.chars
            selected.append(
                {
                    "sub_context": context_row,
                    "selection_score": row["selection_score"],
                    "selection_reasons": row["selection_reasons"],
                }
            )
        return selected

    def _candidate_pool(self, request: RequestProfile) -> list[SubContext]:
        pool: dict[str, SubContext] = {}
        if request.task_id:
            for context in self.registry.by_task(request.task_id):
                pool[context.id] = context
        if request.category:
            for context in self.registry.by_category(request.category):
                pool.setdefault(context.id, context)
        if not pool:
            for context in self.registry.all():
                pool[context.id] = context
        return list(pool.values())

    def _score_context(self, request: RequestProfile, context: SubContext) -> tuple[float, list[str]]:
        query_blob = " ".join([request.query, *request.required_capabilities])
        lexical = self.judge.score_texts(query_blob, " ".join([context.title, context.question_or_objective, context.text]))
        score = lexical["lexical_score"]
        reasons = [f"lexical={lexical['lexical_score']:.3f}"]

        if request.task_id and context.task_id == request.task_id:
            score += 0.25
            reasons.append("same task")
        if request.category and context.category == request.category:
            score += 0.10
            reasons.append("same category")

        capability_overlap = set(request.required_capabilities) & set(context.expected_capability)
        if capability_overlap:
            score += 0.08 * len(capability_overlap)
            reasons.append("capability overlap: " + ", ".join(sorted(capability_overlap)))

        if context.relevance_score is not None:
            score += 0.05 * context.relevance_score
            reasons.append(f"prior relevance={context.relevance_score:.2f}")
        if context.redundancy_risk is not None:
            score -= 0.03 * context.redundancy_risk
            reasons.append(f"redundancy risk={context.redundancy_risk:.2f}")

        return score, reasons
