"""Conservative reuse-decision logic for semantic sub-contexts."""

from __future__ import annotations

from .registry import SubContextRegistry
from .schema import ReuseDecision, SubContext
from .similarity import LexicalSimilarityJudge


class ReuseDecisionEngine:
    """Decide whether two sub-contexts are safe reuse candidates.

    The default policy is deliberately conservative:

    - exact normalized content hash matches are safe to reuse;
    - high lexical similarity is review-required, not automatically reusable;
    - broad topic overlap is rejected by default.

    Embedding and LLM judge outputs can be layered on top of this component in a
    later runtime prototype.
    """

    def __init__(
        self,
        registry: SubContextRegistry,
        judge: LexicalSimilarityJudge | None = None,
        review_threshold: float = 0.55,
        reject_threshold: float = 0.25,
    ) -> None:
        self.registry = registry
        self.judge = judge or LexicalSimilarityJudge()
        self.review_threshold = review_threshold
        self.reject_threshold = reject_threshold

    def decide_pair(self, left: SubContext, right: SubContext) -> ReuseDecision:
        signal = self.judge.compare(left, right)
        if signal.exact_hash_match:
            return ReuseDecision(
                left_id=left.id,
                right_id=right.id,
                decision="safe_exact_content_reuse",
                reason="normalized content hashes match exactly",
                safe_to_reuse=True,
                requires_manual_or_llm_review=False,
                similarity=signal,
            )

        if signal.lexical_score >= self.review_threshold:
            return ReuseDecision(
                left_id=left.id,
                right_id=right.id,
                decision="review_required",
                reason=(
                    "high lexical similarity is not enough for semantic reuse; "
                    "send to embedding/LLM judge and require same-answer utility"
                ),
                safe_to_reuse=False,
                requires_manual_or_llm_review=True,
                similarity=signal,
            )

        if signal.lexical_score >= self.reject_threshold and (signal.same_task or signal.same_category):
            return ReuseDecision(
                left_id=left.id,
                right_id=right.id,
                decision="reject_topic_or_partial_overlap",
                reason="related context detected, but similarity is too weak for reuse",
                safe_to_reuse=False,
                requires_manual_or_llm_review=False,
                similarity=signal,
            )

        return ReuseDecision(
            left_id=left.id,
            right_id=right.id,
            decision="reject_unrelated",
            reason="no safe reuse signal",
            safe_to_reuse=False,
            requires_manual_or_llm_review=False,
            similarity=signal,
        )

    def decide_all_pairs(
        self,
        same_category_only: bool = False,
        different_task_only: bool = False,
        max_pairs: int | None = None,
    ) -> list[ReuseDecision]:
        contexts = self.registry.all()
        decisions: list[ReuseDecision] = []
        for left_index, left in enumerate(contexts):
            for right in contexts[left_index + 1 :]:
                if same_category_only and left.category != right.category:
                    continue
                if different_task_only and left.task_id == right.task_id:
                    continue
                decisions.append(self.decide_pair(left, right))
        decisions.sort(
            key=lambda item: (
                item.safe_to_reuse,
                item.requires_manual_or_llm_review,
                item.similarity.lexical_score,
            ),
            reverse=True,
        )
        return decisions[:max_pairs] if max_pairs else decisions
