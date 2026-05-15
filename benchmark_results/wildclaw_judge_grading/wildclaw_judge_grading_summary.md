# WildClaw Judge Grading Summary

- Source CSV: `benchmark_results/wildclaw_framework_eval_runs/wildclaw_combined_12_with_task4_rerun/wildclaw_framework_eval_results_web_annotated_combined.csv`
- Rows: 12
- Mode: `manual`

## Condition Summary

| condition | rows | avg weighted total | pass | review/fail |
| --- | ---: | ---: | ---: | ---: |
| `candidate_evidence_context` | 3 | 0.967 | 3 | 0 |
| `file_based_chunks` | 3 | 0.983 | 3 | 0 |
| `fixed_size_chunks` | 3 | 0.967 | 3 | 0 |
| `v3_semantic_final_pass` | 3 | 0.975 | 3 | 0 |

## Notes

- This grading separates correctness from context efficiency.
- `manual_reviewer_override` is intentionally blank so a human reviewer can revise scores without changing raw run outputs.
- Use LLM judge mode only as a second reviewer; keep the manual sheet as the auditable artifact.
