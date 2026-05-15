# WildClaw SGLang Runtime Summary

- Source CSV: `benchmark_results/wildclaw_order_permutation_runtime_runs/wildclaw_sglang_runtime_20260515_171823/wildclaw_sglang_runtime_results.csv`
- Rows: 6
- Completed: 6
- Errors: 0

## Condition And Replay Summary

| condition | replay | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| real_semantic_order_original | 1 | 2 | 1039.5 | 75.5 | 964 | 7.3% | 0.4764 |
| real_semantic_order_repeat | 1 | 2 | 1039.5 | 1038.5 | 1 | 99.9% | 0.4013 |
| real_semantic_order_permuted | 1 | 2 | 1039.5 | 408 | 631.5 | 39.2% | 0.4037 |

## Replay-1 Context Reduction Vs Fixed Size

| condition | tasks | fixed prompt sum | condition prompt sum | weighted reduction | mean task reduction |
| --- | --- | --- | --- | --- | --- |

## Notes

- Replay 2 measures ordinary SGLang prefix-cache reuse on identical prompts.
- Order-permutation manifests may encode the repeat as a separate `real_semantic_order_repeat` row with `REPEAT=1`.
- The first real generation can include CUDA/Triton JIT warmup, so latency comparisons should be interpreted with warmup effects in mind.
- `v3_semantic_final_pass` is expected to reduce prompt tokens when the semantic segmenter can isolate task-relevant sub-context.
