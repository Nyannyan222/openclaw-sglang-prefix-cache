# WildClaw Formal Next-Step Execution Plan

## Why These Steps Help

These four workstreams directly support the final goal:

1. Full 12-row WildClaw runtime replay turns the neno5 run from a smoke test
   into a reportable runtime result.
2. Manual / LLM judge grading checks that semantic sub-context selection does
   not reduce answer correctness while reducing context.
3. Mixed-category expansion shows the method is not limited to one Search
   Retrieval case.
4. Real semantic order-permutation testing isolates a native SGLang
   RadixAttention limitation: cache reuse depends on ordered prefixes, even
   when the underlying semantic evidence set is the same.

The fourth step must use real WildClaw semantic sub-contexts. It should not use
toy manually constructed context blocks.

## Generated Artifacts

### Full 12-row runtime replay

- Manifest:
  `benchmark_results/wildclaw_phase2/wildclaw_phase2_sglang_runtime_manifest.jsonl`
- Rows: 12
- Scope: 3 Search Retrieval tasks x 4 context policies

Run on neno5:

```bash
sbatch --account=MST114180 --export=ALL,LIMIT=0,REPEAT=2,MAX_TOKENS=64 scripts/slurm_run_wildclaw_runtime_replay.sh
```

After completion:

```bash
latest_run="$(ls -td benchmark_results/wildclaw_sglang_runtime_runs/wildclaw_sglang_runtime_* | head -1)"
python scripts/summarize_wildclaw_runtime_results.py \
  "$latest_run/wildclaw_sglang_runtime_results.csv" \
  --output "$latest_run/wildclaw_sglang_runtime_summary.md"
cat "$latest_run/wildclaw_sglang_runtime_summary.md"
```

### Judge / manual grading

- Script: `scripts/grade_wildclaw_framework_eval.py`
- Manual seeded sheet:
  `benchmark_results/wildclaw_judge_grading/wildclaw_manual_grading_sheet_seeded.csv`
- Combined grading records:
  `benchmark_results/wildclaw_judge_grading/wildclaw_grading_records.csv`
- Summary:
  `benchmark_results/wildclaw_judge_grading/wildclaw_judge_grading_summary.md`

Manual-first mode:

```bash
python scripts/grade_wildclaw_framework_eval.py --mode manual
```

Optional LLM judge mode:

```bash
OPENAI_API_KEY=... python scripts/grade_wildclaw_framework_eval.py --mode both
```

The manual sheet remains the auditable artifact. LLM judge output should be
treated as a second reviewer, not the only source of truth.

### Mixed-category expansion

- Manifest:
  `benchmark_results/wildclaw_mixed_category_eval/wildclaw_mixed_category_eval_manifest.jsonl`
- Summary:
  `benchmark_results/wildclaw_mixed_category_eval/wildclaw_mixed_category_eval_summary.csv`
- Categories:
  `01_Productivity_Flow`, `04_Search_Retrieval`, `06_Safety_Alignment`
- Rows: 21

Current caveat: local workspace files are available for the Search Retrieval
tasks, while Productivity and Safety rows are prompt-only unless the matching
WildClawBench workspaces are added locally.

### Real semantic order-permutation test

- Script: `scripts/prepare_wildclaw_order_permutation_test.py`
- Manifest:
  `benchmark_results/wildclaw_order_permutation_test/wildclaw_order_permutation_runtime_manifest.jsonl`
- Protocol:
  `benchmark_results/wildclaw_order_permutation_test/wildclaw_order_permutation_protocol.md`
- Rows: 6

Run on neno5:

```bash
sbatch --account=MST114180 --export=ALL,MANIFEST=benchmark_results/wildclaw_order_permutation_test/wildclaw_order_permutation_runtime_manifest.jsonl,LIMIT=0,REPEAT=1,MAX_TOKENS=64,OUTPUT_DIR=benchmark_results/wildclaw_order_permutation_runtime_runs scripts/slurm_run_wildclaw_runtime_replay.sh
```

Interpretation:

- `real_semantic_order_original` establishes the first prompt.
- `real_semantic_order_repeat` repeats the exact same real-context prompt and
  should show high native prefix-cache reuse.
- `real_semantic_order_permuted` uses the same real semantic sub-context set in
  a different order and should show lower cache reuse.

This supports the limitation claim while staying within the WildClaw real
context setting.
