# WildClaw SGLang Runtime Summary

- Source CSV: `benchmark_results/wildclaw_sglang_runtime_runs/wildclaw_sglang_runtime_20260515_171653/wildclaw_sglang_runtime_results.csv`
- Rows: 24
- Completed: 24
- Errors: 0

## Condition And Replay Summary

| condition | replay | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_size_chunks | 1 | 3 | 2808 | 129 | 794 | 3.1% | 0.9630 |
| fixed_size_chunks | 2 | 3 | 2808 | 2807 | 1 | 100.0% | 0.6166 |
| file_based_chunks | 1 | 3 | 2570 | 123.3 | 2446.7 | 4.8% | 0.5968 |
| file_based_chunks | 2 | 3 | 2570 | 2569 | 1 | 100.0% | 0.5510 |
| candidate_evidence_context | 1 | 3 | 2655 | 124 | 2531 | 4.7% | 0.5870 |
| candidate_evidence_context | 2 | 3 | 2655 | 2654 | 1 | 100.0% | 0.5555 |
| v3_semantic_final_pass | 1 | 3 | 915.3 | 125.3 | 790 | 13.7% | 0.5544 |
| v3_semantic_final_pass | 2 | 3 | 915.3 | 914.3 | 1 | 99.9% | 0.5507 |

## Replay-1 Context Reduction Vs Fixed Size

| condition | tasks | fixed prompt sum | condition prompt sum | weighted reduction | mean task reduction |
| --- | --- | --- | --- | --- | --- |
| fixed_size_chunks | 3 | 8424 | 8424 | 0.0% | 0.0% |
| file_based_chunks | 3 | 8424 | 7710 | 8.5% | 3.5% |
| candidate_evidence_context | 3 | 8424 | 7965 | 5.4% | -0.9% |
| v3_semantic_final_pass | 3 | 8424 | 2746 | 67.4% | 32.7% |

## Notes

- Replay 2 measures ordinary SGLang prefix-cache reuse on identical prompts.
- Order-permutation manifests may encode the repeat as a separate `real_semantic_order_repeat` row with `REPEAT=1`.
- The first real generation can include CUDA/Triton JIT warmup, so latency comparisons should be interpreted with warmup effects in mind.
- `v3_semantic_final_pass` is expected to reduce prompt tokens when the semantic segmenter can isolate task-relevant sub-context.
