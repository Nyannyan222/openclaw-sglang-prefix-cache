# WildClawBench Next-Round Framework Experiment

## Goal

Evaluate the framework with real WildClawBench contexts using four context policies:

1. `fixed_size_chunks`
2. `file_based_chunks`
3. `candidate_evidence_context`
4. `v3_semantic_final_pass`

## Prepared Inputs

| Condition | Task Count | Total Chunks | Avg Chunks/Task | Avg Context Chars | Avg Context Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `candidate_evidence_context` | 3 | 7 | 2.33 | 4227.3 | 0.2249 |
| `file_based_chunks` | 3 | 10 | 3.33 | 4050.0 | 0.3924 |
| `fixed_size_chunks` | 3 | 5 | 1.67 | 4716.3 | 0.113 |
| `v3_semantic_final_pass` | 3 | 8 | 2.67 | 1030.7 | 0.387 |

## Final V3 Dataset Quality

- Rows kept: 8
- Avg independence: 5.0 / 5
- Avg relevance: 5.0 / 5
- Avg completeness: 4.125 / 5
- Avg redundancy: 1.0 / 5

## Next Execution Step

Run the framework once per row in `wildclaw_framework_eval_manifest_v3.jsonl`.
For each run, record:

- answer correctness or manual usefulness score,
- cited evidence IDs / source paths,
- prompt/context tokens if available,
- latency and cache metrics if connected to the SGLang benchmark path.

Use `wildclaw_framework_eval_results_template.csv` as the run log.

The expected comparison is whether `v3_semantic_final_pass` preserves evidence
quality while reducing context relative to file-based and fixed-size policies.
