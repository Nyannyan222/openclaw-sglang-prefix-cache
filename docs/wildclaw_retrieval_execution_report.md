# WildClawBench Retrieval/Search Execution Pilot Report

## Goal

This pilot replaces toy synthetic contexts with real WildClawBench task
workspaces and evaluates whether LLM semantic sub-contexts can preserve task
performance while reducing context cost.

The current run focuses on three `04_Search_Retrieval` tasks:

- `task_2_conflicting_handling`: local legal materials conflict with current
  web evidence.
- `task_4_efficient_search`: public web/GitHub evidence search.
- `task_5_fuzzy_search`: fuzzy paper and repository lookup.

## Pipeline

The evaluation pipeline is:

```text
WildClaw task prompt + workspace files
-> real long context extraction
-> candidate evidence extraction
-> LLM semantic sub-context segmentation
-> context-policy manifest
-> retrieval/search execution
-> answer generation
-> lightweight correctness/evidence annotation
```

The retrieval/search execution stage uses the `openai-web` backend in
`scripts/run_wildclaw_framework_eval.py`. It is two-stage:

1. Focused task-specific web retrieval.
2. Final answer generation using local context plus retrieved web evidence.

This two-stage design was needed because direct web-enabled answering sometimes
searched for benchmark/framework terms instead of the actual task evidence.

## Compared Context Policies

| Policy | Description |
| --- | --- |
| `fixed_size_chunks` | Baseline fixed-size chunking over extracted context. |
| `file_based_chunks` | Baseline file-level chunking. |
| `candidate_evidence_context` | Keyword/evidence selected candidate context before LLM segmentation. |
| `v3_semantic_final_pass` | Reviewed pass-only LLM semantic sub-contexts from V3. |

## Main Result

Combined run:

```text
benchmark_results/wildclaw_framework_eval_runs/wildclaw_combined_12_with_task4_rerun/
```

Task 4 was rerun after tuning the search brief toward official Python
documentation. The combined report replaces the original partial task 4 rows
with the tuned rerun.

| condition | rows | avg correctness | avg evidence | avg context reduction | avg prompt tokens |
|---|---:|---:|---:|---:|---:|
| candidate_evidence_context | 3 | 1.0 | 5.0 | 0.225 | 4250.7 |
| file_based_chunks | 3 | 1.0 | 5.0 | 0.392 | 4175.7 |
| fixed_size_chunks | 3 | 1.0 | 5.0 | 0.113 | 4452.3 |
| v3_semantic_final_pass | 3 | 1.0 | 5.0 | 0.387 | 2169.3 |

## Interpretation

All four policies solved the three pilot tasks after retrieval/search execution
was added. The strongest result is efficiency: `v3_semantic_final_pass`
maintained the same lightweight correctness score while using far fewer prompt
tokens.

Compared with baselines, `v3_semantic_final_pass` used:

- About 51% fewer prompt tokens than `fixed_size_chunks`.
- About 48% fewer prompt tokens than `file_based_chunks`.
- About 49% fewer prompt tokens than `candidate_evidence_context`.

This supports the current hypothesis: real-workspace semantic sub-contexts can
preserve the necessary evidence while reducing context cost.

## Task-Level Notes

| Task | Outcome |
| --- | --- |
| `task_2_conflicting_handling` | Correctly resolved local outdated two-year limitation evidence against current Civil Code three-year evidence. |
| `task_4_efficient_search` | Correct after query tuning; official Python documentation evidence gives Python `3.12` and `gh-90385`. |
| `task_5_fuzzy_search` | Correctly identified `Visual-RFT`, Liu, and repository star evidence. |

## Caveats

- The pilot is small: only 3 Search Retrieval tasks and 12 evaluated rows.
- Correctness is currently lightweight heuristic annotation, not a full human or
  LLM-judge grading protocol.
- Some tasks are web-dominant, so local context reduction can be less meaningful
  than prompt-token cost.
- Query quality matters: task 4 initially returned partial results until the
  search brief was tuned toward official sources.

## Recommended Next Step

Expand from this Search Retrieval pilot to a mixed-category evaluation:

1. Add 2-3 `Productivity` tasks.
2. Add 2-3 `Safety` tasks.
3. Reuse the same four context policies.
4. Keep the two-stage retrieval/search execution backend for tasks requiring
   public/current evidence.
5. Add a small manual grading sheet for answer correctness and evidence
   sufficiency.

The next report should compare whether semantic sub-contexts still reduce token
cost when tasks require workspace operations, safety constraints, or less web
retrieval.
