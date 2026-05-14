# WildClaw Similarity Task Selector

This selector chooses WildClawBench tasks/workspaces that are most likely to
produce genuinely semantically similar sub-contexts in the next discovery run.
It is a pre-filter, not the final semantic judge.

## Goal

The strict semantic discovery pipeline already showed that small pilots mostly
contain broad-topic or partial-overlap pairs. The selector helps find better
source tasks by ranking workspaces with signals such as:

- real local workspace availability
- multiple usable workspace files after answer/grading leakage filters
- exact duplicate files
- exact duplicate paragraph/window segments across files
- near-duplicate paragraph/window segments across files
- domain hints for law, policy, safety, instructions, API docs, and code docs

The output should be used as the input list for `real_context_extractor.py`,
then `llm_semantic_segmenter.py`, then
`find_semantic_similar_subcontexts.py`.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\select_wildclaw_similarity_tasks.py `
  --wildclaw-root external\WildClawBench `
  --output-dir benchmark_results\wildclaw_similarity_task_selection `
  --select-top 8 `
  --min-select-score 1
```

Useful options:

- `--category 04_Search_Retrieval` restricts the scan to one category.
- `--near-duplicate-threshold 0.35` relaxes the local near-duplicate heuristic.
- `--min-select-score 0` keeps weaker candidates in the selected task list.
- `--max-chars-per-file` and `--max-file-bytes` control local workspace reading.

## Outputs

The selector writes:

- `wildclaw_similarity_task_candidates.csv`
- `wildclaw_similarity_task_candidates.json`
- `wildclaw_similarity_selected_tasks.txt`
- `wildclaw_similarity_task_selection_report.md`

Current local run:

- Tasks scanned: 60
- Positive selected tasks: 1
- Selected task: `04_Search_Retrieval_task_2_conflicting_handling`
- Reason: it is the only locally available workspace with substantial usable
  text in this checkout: 14 included files, 276,236 included characters, and
  148 text segments after filtering.

Other categories currently rank low because their workspace directories are not
present in the local WildClawBench checkout. If a fuller WildClawBench
workspace is synced later, rerun this selector before expanding semantic
segmentation.

## Next Step

Use the selected task list to build the next focused pilot:

```powershell
.\.venv\Scripts\python.exe scripts\real_context_extractor.py `
  --wildclaw-root external\WildClawBench `
  --output-dir benchmark_results\wildclaw_similarity_selected_pilot `
  --task 04_Search_Retrieval_task_2_conflicting_handling
```

Then run semantic segmentation and strict semantic similarity discovery on
`benchmark_results\wildclaw_similarity_selected_pilot`.
