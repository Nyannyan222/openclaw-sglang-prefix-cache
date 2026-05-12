# WildClawBench Real-Context Semantic Sub-Context Pilot

## Goal

Use WildClawBench real task workspaces as the evaluation source and replace the
previous synthetic A/B/C setup for the next stage of the sub-context work. The
first phase does not run the complete WildClawBench benchmark. It builds a
small pilot dataset from real task prompts and workspace files, then compares
three context construction strategies:

| Method | Purpose |
| --- | --- |
| Fixed-size chunking | Baseline window slicing by character count. |
| File-based chunking | Baseline that treats each prompt/file as one context unit. |
| LLM semantic sub-context | Proposed method: semantically independent sub-contexts. |

## Pilot Scope

Start with `04_Search_Retrieval` because it directly exercises retrieval,
evidence selection, conflict handling, and search constraints. The first three
pilot tasks are:

| Task | Why it is useful for the pilot |
| --- | --- |
| `04_Search_Retrieval_task_2_conflicting_handling` | Local legal files may conflict with up-to-date web evidence, so it tests conflict resolution and source freshness. |
| `04_Search_Retrieval_task_4_efficient_search` | Has an explicit search budget and evidence-chain requirement, so it tests compact context extraction under constraints. |
| `04_Search_Retrieval_task_5_fuzzy_search` | Requires fuzzy intent reconstruction from partial memory, so it tests semantic relevance rather than exact keyword matching. |

## Data Inspection Notes

WildClawBench provides 60 tasks across six categories. Its Hugging Face
workspace contains initial task files, while the GitHub task markdown files
contain prompts, expected behavior, grading code, workspace paths, skills, and
environment requirements. For this pilot, the extractor uses only:

- the task prompt,
- non-ground-truth workspace files under the task workspace,
- file metadata for unsupported binary files.

The extractor excludes:

- `gt/`,
- grader or score files,
- answer-like filenames,
- generated `results/` or `output/` directories.

This keeps the pilot focused on real task context without leaking evaluation
answers.

## Commands

Download the lightweight parts needed for the pilot:

```bash
git clone https://github.com/InternLM/WildClawBench external/WildClawBench
huggingface-cli download internlm/WildClawBench \
  --repo-type dataset \
  --include "workspace/04_Search_Retrieval/**" \
  --local-dir external/WildClawBench
```

Extract long contexts for the three pilot tasks:

```bash
python scripts/real_context_extractor.py \
  --wildclaw-root external/WildClawBench \
  --output-dir benchmark_results/wildclaw_semantic_subcontext_pilot
```

Run LLM semantic segmentation. With `OPENROUTER_API_KEY` or `OPENAI_API_KEY`
set, `--mode auto` uses the LLM path. Without an API key, it writes a heuristic
dry-run so the rest of the pipeline can be checked.

```bash
python scripts/llm_semantic_segmenter.py \
  benchmark_results/wildclaw_semantic_subcontext_pilot/wildclaw_real_contexts_pilot.jsonl \
  --output-dir benchmark_results/wildclaw_semantic_subcontext_pilot \
  --mode auto
```

Expected outputs:

```text
benchmark_results/wildclaw_semantic_subcontext_pilot/
  wildclaw_data_inspection.json
  wildclaw_real_contexts_pilot.jsonl
  wildclaw_semantic_subcontext_pilot.jsonl
  wildclaw_fixed_size_chunks_pilot.jsonl
  wildclaw_file_based_chunks_pilot.jsonl
  wildclaw_chunking_comparison.csv
  wildclaw_manual_review_sheet.csv
  wildclaw_semantic_segmenter_run.json
```

## Manual Review Protocol

Manually inspect 20-30 rows using `wildclaw_manual_review_sheet.csv`, with
full text available in `wildclaw_semantic_subcontext_pilot.jsonl`. For each
row, check:

| Field | Question |
| --- | --- |
| Independence | Can this sub-context be understood without the full original task context? |
| Relevance | Is it useful for the task objective? |
| Completeness | Does it preserve enough evidence for the atomic objective? |
| Redundancy | Is it mostly duplicate material already captured by another sub-context? |
| Leakage | Does it avoid ground truth, grader answers, and generated result files? |

Suggested manual labels:

```text
manual_review_status = pass | revise | reject
manual_independence = 1-5
manual_relevance = 1-5
manual_completeness = 1-5
manual_redundancy = 1-5
review_notes = short reason
```

## First-Week Deliverables

| # | Deliverable | Status |
| --- | --- | --- |
| 1 | Complete WildClawBench data inspection | Ready through `wildclaw_data_inspection.json`. |
| 2 | Select 3 `04_Search_Retrieval` tasks for pilot | Selected: tasks 2, 4, and 5. |
| 3 | Write `real_context_extractor.py` | Done. |
| 4 | Write `llm_semantic_segmenter.py` | Done. |
| 5 | Produce `wildclaw_semantic_subcontext_pilot.jsonl` | Produced after running segmentation command. |
| 6 | Manually inspect 20-30 sub-contexts | Pending human review. |
| 7 | Compare chunking methods | Produced as `wildclaw_chunking_comparison.csv`. |

## Report Table Template

| Method | Independence | Relevance | Completeness | Redundancy Risk | Context Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-size chunking | from CSV | from CSV | from CSV | from CSV | from CSV |
| File-based chunking | from CSV | from CSV | from CSV | from CSV | from CSV |
| LLM semantic sub-context | from CSV + manual review | from CSV + manual review | from CSV + manual review | from CSV + manual review | from CSV |

The initial CSV scores are automatic estimates. The report should treat the
manual labels as the authoritative quality check for the first 20-30
sub-contexts.

## Retrieval/Search Execution Stage

The next-round evaluation now includes a two-stage `openai-web` backend in
`scripts/run_wildclaw_framework_eval.py`:

1. Focused web retrieval with a short task-specific search prompt.
2. Final framework answer using both the local context policy and retrieved web
   evidence.

This is important for `04_Search_Retrieval` because some tasks require current
law, public webpages, GitHub evidence, or paper/repository evidence that cannot
be solved from local chunks alone.

Example full run:

```bash
python scripts/run_wildclaw_framework_eval.py \
  --backend openai-web \
  --model gpt-4o-mini \
  --manifest benchmark_results/wildclaw_next_experiment/wildclaw_framework_eval_manifest_v3.jsonl \
  --output-dir benchmark_results/wildclaw_framework_eval_runs \
  --max-tokens 900 \
  --web-max-tokens 700 \
  --web-tool-choice required
```

Then annotate the run:

```bash
python scripts/annotate_wildclaw_framework_eval.py \
  benchmark_results/wildclaw_framework_eval_runs/<run_dir>
```

Current retrieval/search pilot run:

```text
benchmark_results/wildclaw_framework_eval_runs/wildclaw_framework_eval_20260512_221654/
  retrieval/*.web.md
  answers/*.md
  wildclaw_framework_eval_results_web_annotated.csv
  wildclaw_framework_eval_condition_summary_web_annotated.csv
  run_summary_web.md
```

Observed first-pass result: task 2 and task 5 are solved after adding web
retrieval. Task 4 is partial because the run found Python 3.12 but did not
consistently confirm the official GitHub reference for `pathlib.Path.walk()`;
that task should be used as the next query-tuning case.

Task 4 query tuning was then rerun with an official-docs-first search brief:

```text
benchmark_results/wildclaw_framework_eval_runs/wildclaw_framework_eval_20260512_224959/
```

The tuned run solved all four task 4 conditions by retrieving the Python 3.12
What's New evidence for `pathlib.Path.walk()` and `gh-90385`. A combined
12-row report replacing the original task 4 rows is available at:

```text
benchmark_results/wildclaw_framework_eval_runs/wildclaw_combined_12_with_task4_rerun/run_summary_web_combined.md
```
