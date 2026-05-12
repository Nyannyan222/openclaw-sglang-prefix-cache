# WildClaw Retrieval/Search Execution Run Summary

This run uses a two-stage `openai-web` backend: focused web retrieval first, final answer second.

## Condition Summary

| condition | rows | avg correctness | avg evidence | avg context reduction | avg prompt tokens |
|---|---:|---:|---:|---:|---:|
| candidate_evidence_context | 1 | 1.0 | 5.0 | -0.072 | 2766.0 |
| file_based_chunks | 1 | 1.0 | 5.0 | 0.054 | 2681.0 |
| fixed_size_chunks | 1 | 1.0 | 5.0 | -0.157 | 2647.0 |
| v3_semantic_final_pass | 1 | 1.0 | 5.0 | -0.085 | 2647.0 |

## Task Notes

- `04_Search_Retrieval_task_4_efficient_search`: avg correctness 1.000. Has Python 3.12 and official gh-90385 evidence.
