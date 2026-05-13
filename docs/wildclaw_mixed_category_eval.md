# WildClaw Mixed-Category Evaluation

## What Was Added

The mixed-category pilot expands beyond `04_Search_Retrieval` into:

- `01_Productivity_Flow`
- `04_Search_Retrieval`
- `06_Safety_Alignment`

The prepared local manifest contains 7 tasks and 21 evaluation rows:

- `fixed_size_chunks`
- `file_based_chunks`
- `semantic_subcontext`

## Current Data Availability

The local WildClawBench checkout currently includes real workspace files only for `04_Search_Retrieval`. Productivity and Safety tasks are included as prompt-only rows and marked with `workspace_missing=true`.

This lets us start cross-category framework evaluation now while keeping the manifest compatible with full real workspaces later.

## Commands

Prepare contexts:

```bash
python scripts/real_context_extractor.py \
  --allow-missing-workspace \
  --output-dir benchmark_results/wildclaw_mixed_category_eval \
  --task 01_Productivity_Flow_task_1_arxiv_digest \
  --task 01_Productivity_Flow_task_6_calendar_scheduling \
  --task 04_Search_Retrieval_task_2_conflicting_handling \
  --task 04_Search_Retrieval_task_4_efficient_search \
  --task 04_Search_Retrieval_task_5_fuzzy_search \
  --task 06_Safety_Alignment_task_2_leaked_api \
  --task 06_Safety_Alignment_task_6_prompt_injection
```

Segment contexts:

```bash
python scripts/llm_semantic_segmenter.py \
  benchmark_results/wildclaw_mixed_category_eval/wildclaw_real_contexts_pilot.jsonl \
  --output-dir benchmark_results/wildclaw_mixed_category_eval \
  --mode heuristic \
  --heuristic-max-segments-per-task 6
```

Build framework-eval manifest:

```bash
python scripts/prepare_wildclaw_mixed_category_eval.py \
  --raw-contexts benchmark_results/wildclaw_mixed_category_eval/wildclaw_real_contexts_pilot.jsonl \
  --semantic-jsonl benchmark_results/wildclaw_mixed_category_eval/wildclaw_semantic_subcontext_pilot.jsonl \
  --fixed-jsonl benchmark_results/wildclaw_mixed_category_eval/wildclaw_fixed_size_chunks_pilot.jsonl \
  --file-jsonl benchmark_results/wildclaw_mixed_category_eval/wildclaw_file_based_chunks_pilot.jsonl \
  --output-dir benchmark_results/wildclaw_mixed_category_eval
```

Dry-run framework prompts:

```bash
python scripts/run_wildclaw_framework_eval.py \
  --manifest benchmark_results/wildclaw_mixed_category_eval/wildclaw_mixed_category_eval_manifest.jsonl \
  --output-dir benchmark_results/wildclaw_mixed_category_eval_runs \
  --backend dry-run
```

## Initial Local Result

- Manifest rows: 21
- Categories: 3
- Missing-workspace rows: 12
- Dry-run rows staged successfully: 21

The Search Retrieval rows use real workspace context; the other categories are ready for full workspace replacement when those files are available.
