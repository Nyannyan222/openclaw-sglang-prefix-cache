# NOC Semantic Sub-context Manager Smoke Run

## Registry Summary

- Sub-contexts: 10
- Tasks: 3
- Categories: 1
- Duplicate content hashes: 0

## Relevant vs Reusable

The selected context list answers the relevance question: which sub-contexts
should accompany this request. The reuse relation table answers a separate
question: whether two sub-contexts are safe to reuse as equivalent evidence.

| relation type | meaning | reuse eligibility |
| --- | --- | --- |
| `exact_duplicate` | content is nearly identical after normalization | yes |
| `near_duplicate` | different wording but potentially equivalent information | yes, after judge |
| `same_answer_utility` | either context can support the same answer/evidence role | yes |
| `partial_overlap` | some information overlaps, but evidence role may differ | no / maybe |
| `broad_topic` | same topic/domain but different use | no |
| `unrelated` | no meaningful relation | no |

## Selected Contexts

| rank | score | sub-context | task | title | chars |
| ---: | ---: | --- | --- | --- | ---: |
| 1 | 0.7999 | `wildclaw_04_Search_Retrieval_task_2_conflicting_handling_llm_semantic_subctx_01` | `04_Search_Retrieval_task_2_conflicting_handling` | Statute of Limitations for Civil Claims | 125 |
| 2 | 0.7858 | `wildclaw_04_Search_Retrieval_task_2_conflicting_handling_llm_semantic_subctx_05` | `04_Search_Retrieval_task_2_conflicting_handling` | Interruption and Suspension of Statute of Limitations | 75 |
| 3 | 0.6052 | `wildclaw_04_Search_Retrieval_task_2_conflicting_handling_llm_semantic_subctx_06` | `04_Search_Retrieval_task_2_conflicting_handling` | Debt Recovery Procedures | 68 |
| 4 | 0.5908 | `wildclaw_04_Search_Retrieval_task_2_conflicting_handling_llm_semantic_subctx_03` | `04_Search_Retrieval_task_2_conflicting_handling` | Conditions for Civil Rights and Obligations | 71 |
| 5 | 0.5388 | `wildclaw_04_Search_Retrieval_task_2_conflicting_handling_llm_semantic_subctx_04` | `04_Search_Retrieval_task_2_conflicting_handling` | Legal Framework for Civil Rights | 84 |
| 6 | 0.5307 | `wildclaw_04_Search_Retrieval_task_2_conflicting_handling_llm_semantic_subctx_02` | `04_Search_Retrieval_task_2_conflicting_handling` | Civil Liability for Breach of Contract | 101 |

## Reuse Relation Summary

| relation type | count |
| --- | ---: |
| `broad_topic` | 45 |

## Decision Summary

| decision | count |
| --- | ---: |
| `do_not_reuse` | 45 |

## Interpretation

This prototype separates relevance from reusability. Embedding or lexical
similarity can find candidates, but the reuse decision remains conservative.
Only exact duplicates are immediately reusable. Near duplicates require
embedding/LLM judge confirmation, and same-answer utility should be required
before treating two sub-contexts as reusable evidence.
