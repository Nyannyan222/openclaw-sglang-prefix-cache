# Combined WildClaw Retrieval/Search Evaluation Summary

Base run: `wildclaw_framework_eval_20260512_221654`
Task-specific rerun: `wildclaw_framework_eval_20260512_224959`

| condition | rows | avg correctness | avg evidence | avg context reduction | avg prompt tokens |
|---|---:|---:|---:|---:|---:|
| candidate_evidence_context | 3 | 1.0 | 5.0 | 0.225 | 4250.7 |
| file_based_chunks | 3 | 1.0 | 5.0 | 0.392 | 4175.7 |
| fixed_size_chunks | 3 | 1.0 | 5.0 | 0.113 | 4452.3 |
| v3_semantic_final_pass | 3 | 1.0 | 5.0 | 0.387 | 2169.3 |

All three pilot tasks now score 1.0 under the lightweight heuristic after replacing the base task 4 rows with the tuned retrieval rerun.
