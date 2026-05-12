# WildClawBench Semantic Sub-Context Pilot V2 Summary

V2 changes the segmentation prompt from short task decomposition to
evidence-bearing semantic sub-context extraction.

## LLM Run

| Task | V2 Sub-Contexts | Error |
| --- | ---: | --- |
| `04_Search_Retrieval_task_2_conflicting_handling` | 6 | none |
| `04_Search_Retrieval_task_4_efficient_search` | 2 | none |
| `04_Search_Retrieval_task_5_fuzzy_search` | 1 | none |
| Total | 9 | none |

Model:

```text
gpt-4o-mini
```

## Automatic Comparison

| Method | Sub-Context Count | Avg Context Reduction |
| --- | ---: | ---: |
| File-based chunking | 17 | 0.8549 |
| Fixed-size chunking | 80 | 0.9611 |
| LLM semantic sub-context V2 | 9 | 0.8764 |

## Manual Review

| Status | Count |
| --- | ---: |
| Pass | 5 |
| Revise | 4 |
| Reject | 0 |

| Metric | V1 Manual Avg | V2 Manual Avg |
| --- | ---: | ---: |
| Independence | 4.71 / 5 | 4.67 / 5 |
| Relevance | 3.92 / 5 | 4.67 / 5 |
| Completeness | 2.58 / 5 | 3.33 / 5 |
| Redundancy | 1.96 / 5 | 1.33 / 5 |

## Interpretation

V2 is qualitatively better than V1:

- fewer metadata-only fragments,
- stronger relevance,
- better completeness,
- lower redundancy.

But V2 is still not enough as the final dataset because it does not extract the
actual legal evidence from local law files for task 2. It mostly captures case
facts and task constraints. The next improvement should add a candidate evidence
retrieval stage before LLM segmentation.
