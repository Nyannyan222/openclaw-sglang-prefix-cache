# WildClaw Retrieval/Search Execution Run Summary

This run uses a two-stage `openai-web` backend: focused web retrieval first, final answer second.

## Condition Summary

| condition | rows | avg correctness | avg evidence | avg context reduction | avg prompt tokens |
|---|---:|---:|---:|---:|---:|
| candidate_evidence_context | 3 | 0.833 | 4.333 | 0.225 | 4160.0 |
| file_based_chunks | 3 | 0.833 | 4.333 | 0.392 | 4104.7 |
| fixed_size_chunks | 3 | 0.833 | 4.333 | 0.113 | 4398.7 |
| v3_semantic_final_pass | 3 | 0.833 | 4.333 | 0.387 | 2113.7 |

## Task Notes

- `04_Search_Retrieval_task_2_conflicting_handling`: avg correctness 1.000. Correct current limitation period; stronger if it explains local two-year law is outdated.
- `04_Search_Retrieval_task_4_efficient_search`: avg correctness 0.500. Partial: has Python 3.12 but did not confirm the official gh-90385 reference. Partial: has Python 3.12 but did not confirm the official gh-90385 reference. It reports #119573, which appears to be unrelated follow-up work.
- `04_Search_Retrieval_task_5_fuzzy_search`: avg correctness 1.000. Identifies Visual-RFT, Liu, and >2k GitHub stars.
