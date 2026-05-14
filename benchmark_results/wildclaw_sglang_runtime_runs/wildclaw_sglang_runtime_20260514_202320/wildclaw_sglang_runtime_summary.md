# WildClaw SGLang Runtime Summary

- Source CSV: `C:/Users/Administrator/OneDrive/文件/New project/benchmark_results/wildclaw_sglang_runtime_runs/wildclaw_sglang_runtime_20260514_202320/wildclaw_sglang_runtime_results.csv`
- Rows: 20
- Completed: 20
- Errors: 0

## Condition And Replay Summary

| condition | replay | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_context | 1 | 5 | 189 | 73.6 | 115.4 | 38.9% | 0.7757 |
| similar_context | 1 | 5 | 163 | 77.8 | 85.2 | 47.7% | 0.6867 |
| canonical_plus_delta | 1 | 5 | 305 | 183 | 122 | 60.0% | 0.6545 |
| canonical_context_repeat | 1 | 5 | 190 | 162 | 28 | 85.3% | 0.6322 |

## Replay-1 Context Reduction Vs Fixed Size

| condition | tasks | fixed prompt sum | condition prompt sum | weighted reduction | mean task reduction |
| --- | --- | --- | --- | --- | --- |

## Notes

- Replay 2 measures ordinary SGLang prefix-cache reuse on identical prompts.
- The first real generation can include CUDA/Triton JIT warmup, so latency comparisons should be interpreted with warmup effects in mind.
- `v3_semantic_final_pass` is expected to reduce prompt tokens when the semantic segmenter can isolate task-relevant sub-context.
