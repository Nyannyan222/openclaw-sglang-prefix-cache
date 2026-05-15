# WildClaw Real Semantic Order-Permutation Test

## Purpose

Show the limitation of native SGLang RadixAttention prefix cache using real WildClaw semantic sub-contexts.

The test sends, per task, three prompts:

1. `real_semantic_order_original`: real semantic sub-contexts in one stable order.
2. `real_semantic_order_repeat`: exact repeat of the original prompt, expected to show high prefix reuse.
3. `real_semantic_order_permuted`: same real sub-context set in a different order, expected to show lower prefix reuse.

## Runtime Command

```bash
sbatch --account=MST114180 --export=ALL,MANIFEST=benchmark_results/wildclaw_order_permutation_test/wildclaw_order_permutation_runtime_manifest.jsonl,LIMIT=0,REPEAT=1,MAX_TOKENS=64,OUTPUT_DIR=benchmark_results/wildclaw_order_permutation_runtime_runs scripts/slurm_run_wildclaw_runtime_replay.sh
```

Use `REPEAT=1` because the manifest already includes the explicit repeat row.

## Scope

- Tasks: 2
- Manifest rows: 6
- Source: WildClawBench real task prompts and LLM semantic sub-context JSONL.

## Interpretation

The important comparison is within each task group:

- Original -> repeat should have high cached-token ratio.
- Original -> permuted should reuse only the shared prompt header and any unchanged leading text.
- If the same semantic evidence appears in a different order but cache reuse drops, this supports the claim that prefix cache is order-sensitive.
