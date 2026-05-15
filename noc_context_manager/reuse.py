"""Conservative reuse-decision logic for semantic sub-contexts."""

from __future__ import annotations

from .registry import SubContextRegistry
from .schema import ReuseDecision, SubContext
from .similarity import LexicalSimilarityJudge


class ReuseDecisionEngine:
    """Decide whether two sub-contexts are safe reuse candidates.

    The default policy is deliberately conservative:

    - exact normalized content hash matches are reusable;
    - near-duplicate candidates require embedding/LLM judge confirmation;
    - same-answer-utility can be ingested later from an LLM judge;
    - partial overlap, broad topic, and unrelated pairs are not reusable.

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
                relation_type="exact_duplicate",
                reuse_eligibility="yes",
                decision="reuse_allowed",
                reason="normalized content hashes match exactly",
                safe_to_reuse=True,
                requires_manual_or_llm_review=False,
                similarity=signal,
            )

        if signal.lexical_score >= self.review_threshold:
            return ReuseDecision(
                left_id=left.id,
                right_id=right.id,
                relation_type="near_duplicate",
                reuse_eligibility="yes_after_judge",
                decision="judge_required",
                reason=(
                    "candidate may be information-equivalent, but lexical similarity alone "
                    "is not enough; send to embedding/LLM judge and require same-answer utility"
                ),
                safe_to_reuse=False,
                requires_manual_or_llm_review=True,
                similarity=signal,
            )

        if signal.lexical_score >= self.reject_threshold and (signal.same_task or signal.same_category):
            return ReuseDecision(
                left_id=left.id,
                right_id=right.id,
                relation_type="partial_overlap",
                reuse_eligibility="no_or_maybe",
                decision="do_not_reuse",
                reason="some information may overlap, but the pair is not equivalent evidence",
                safe_to_reuse=False,
                requires_manual_or_llm_review=False,
                similarity=signal,
            )

        if signal.same_task or signal.same_category:
            return ReuseDecision(
                left_id=left.id,
                right_id=right.id,
                relation_type="broad_topic",
                reuse_eligibility="no",
                decision="do_not_reuse",
                reason="same task/category signal only; topic similarity is not reuse eligibility",
                safe_to_reuse=False,
                requires_manual_or_llm_review=False,
                similarity=signal,
            )

        return ReuseDecision(
            left_id=left.id,
            right_id=right.id,
            relation_type="unrelated",
            reuse_eligibility="no",
            decision="do_not_reuse",
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
