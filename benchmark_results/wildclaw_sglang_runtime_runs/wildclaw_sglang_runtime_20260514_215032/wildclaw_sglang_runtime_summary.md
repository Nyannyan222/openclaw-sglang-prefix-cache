# WildClaw SGLang Runtime Summary

- Source CSV: `benchmark_results/wildclaw_sglang_runtime_runs/wildclaw_sglang_runtime_20260514_215032/wildclaw_sglang_runtime_results.csv`
- Rows: 20
- Completed: 20
- Errors: 0

## Condition And Replay Summary

| condition | replay | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_context | 1 | 5 | 189 | 0 | 189 | 0.0% | 0.7134 |
| similar_context | 1 | 5 | 163 | 28.4 | 134.6 | 17.4% | 0.6608 |
| canonical_plus_delta | 1 | 5 | 305 | 183 | 122 | 60.0% | 0.6836 |
| canonical_context_repeat | 1 | 5 | 190 | 162 | 28 | 85.3% | 0.6733 |

## Replay-1 Context Reduction Vs Fixed Size

| condition | tasks | fixed prompt sum | condition prompt sum | weighted reduction | mean task reduction |
| --- | --- | --- | --- | --- | --- |

## Notes

- Replay 2 measures ordinary SGLang prefix-cache reuse on identical prompts.
- The first real generation can include CUDA/Triton JIT warmup, so latency comparisons should be interpreted with warmup effects in mind.
- `v3_semantic_final_pass` is expected to reduce prompt tokens when the semantic segmenter can isolate task-relevant sub-context.
