# WildClawBench Phase 2 Plan

## Scope

Phase 2 expands the pilot from Search Retrieval to a mixed-category setup:

- `01_Productivity_Flow`: 3 tasks
- `04_Search_Retrieval`: 3 tasks
- `06_Safety_Alignment`: 3 tasks

## Four Workstreams

1. Expand to more WildClawBench categories.
2. Establish formal manual / LLM-judge grading.
3. Connect framework prompts back to SGLang runtime to measure cached tokens, prefill tokens, and latency.
4. Implement runtime-level semantic sub-context KV reuse.

## Current Blocking Item

The local WildClawBench checkout currently lacks workspace directories for:

- `01_Productivity_Flow_task_1_arxiv_digest`
- `01_Productivity_Flow_task_6_calendar_scheduling`
- `01_Productivity_Flow_task_7_openmmlab_contributors`
- `06_Safety_Alignment_task_2_leaked_api`
- `06_Safety_Alignment_task_6_prompt_injection`
- `06_Safety_Alignment_task_8_malicious_comments`

These tasks are selected and ready, but real-context extraction requires the matching workspaces.

## Runtime Metrics

For each runtime row, collect:

- `cached_tokens`
- `prompt_tokens`
- `prefill_tokens` or equivalent prefill-time/token metric
- `time_to_first_token_ms`
- `latency_s`
- `cache_hit_rate`

Runtime replay command:

```bash
python scripts/run_wildclaw_sglang_runtime_replay.py \
  --manifest benchmark_results/wildclaw_phase2/wildclaw_phase2_sglang_runtime_manifest.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model <served-model-name> \
  --repeat 2
```

The first replay approximates cold/prefix-cache-miss behavior. The second
identical replay measures ordinary SGLang prefix-cache reuse. This is the
runtime metric baseline before implementing semantic sub-context KV reuse.

## KV Reuse Design

The first implementation target is an offline replay protocol:

1. Segment each prompt into stable semantic sub-context blocks.
2. Hash each block after canonical serialization.
3. Replay requests in orders that share and reorder blocks.
4. Compare normal prefix cache vs semantic sub-context block reuse.
5. Report token reuse, TTFT, and latency differences.

This keeps the experiment measurable before patching deeper SGLang internals.

## Existing Eval Rows

- Existing annotated rows available for runtime replay: 12
