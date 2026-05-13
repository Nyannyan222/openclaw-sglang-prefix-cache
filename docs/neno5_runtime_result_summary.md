# neno5 WildClaw SGLang Runtime Result Summary

## Run

- Source run: `benchmark_results/wildclaw_sglang_runtime_runs/wildclaw_sglang_runtime_20260513_162149`
- Rows: 24
- Errors: 0
- Setup: WildClawBench-derived real-context replay through SGLang on neno5
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Conditions: `fixed_size_chunks`, `file_based_chunks`, `candidate_evidence_context`, `v3_semantic_final_pass`
- Replays: 2 identical prompt replays per condition

## Main Finding

The neno5 real-runtime replay succeeded. All 24 requests completed without errors, and SGLang prefix-cache reuse was visible on replay 2. For identical prompt replays, cached tokens were nearly equal to prompt tokens, leaving only about 1 prefill token in most replay-2 rows.

The clearest context-reduction result appeared in the long-context task:

| condition | replay 1 prompt tokens | replay 2 cached tokens | replay 2 prefill tokens |
| --- | ---: | ---: | ---: |
| fixed_size_chunks | 6578 | 6577 | 1 |
| file_based_chunks | 5844 | 5843 | 1 |
| candidate_evidence_context | 5982 | 5981 | 1 |
| v3_semantic_final_pass | 984 | 983 | 1 |

For this task, `v3_semantic_final_pass` reduced prompt tokens from 6578 to 984 versus fixed-size chunking, which is about an 85.0% reduction.

## Interpretation

- SGLang prefix cache is working: replay 2 shows near-complete cache hits across conditions.
- Semantic sub-context can substantially reduce context size on long-context WildClaw tasks.
- The first request latency should be treated as warmup-sensitive because CUDA/Triton JIT can affect early requests.
- Runtime-level KV reuse is not yet implemented; this result measures ordinary SGLang prefix-cache reuse over the generated WildClaw replay prompts.

## Next Experiment

Run the full 12-row pilot with the updated SLURM job and use the generated `wildclaw_sglang_runtime_summary.md` for reporting:

```bash
git pull origin main
sbatch --account=MST114180 --export=ALL,LIMIT=12,REPEAT=2 scripts/slurm_run_wildclaw_runtime_replay.sh
```
