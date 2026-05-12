# WildClawBench Semantic Sub-Context Pilot V3 Summary

V3 adds a candidate evidence retrieval stage before LLM semantic segmentation.
The pipeline is:

```text
WildClawBench task prompt + workspace files
-> real long context extraction
-> candidate evidence retrieval
-> LLM semantic sub-context extraction
-> manual quality review
```

## Candidate Evidence Extraction

| Task | Candidate Evidence Passages | Candidate Context Chars |
| --- | ---: | ---: |
| `04_Search_Retrieval_task_2_conflicting_handling` | 7 | 10396 |
| `04_Search_Retrieval_task_4_efficient_search` | 0 | 1660 |
| `04_Search_Retrieval_task_5_fuzzy_search` | 0 | 626 |

For task 2, the tightened evidence filter selected passages from:

```text
exec/04_Search_Retrieval_task_2_conflicting_handling/laws/law12.pdf
```

## LLM Segmentation

| Task | V3 Sub-Contexts | Error |
| --- | ---: | --- |
| `04_Search_Retrieval_task_2_conflicting_handling` | 6 | none |
| `04_Search_Retrieval_task_4_efficient_search` | 3 | none |
| `04_Search_Retrieval_task_5_fuzzy_search` | 1 | none |
| Total | 10 | none |

## Manual Review

| Status | Count |
| --- | ---: |
| Pass | 8 |
| Revise | 1 |
| Reject | 1 |

| Metric | V1 Manual Avg | V2 Manual Avg | V3 Manual Avg |
| --- | ---: | ---: | ---: |
| Independence | 4.71 / 5 | 4.67 / 5 | 4.80 / 5 |
| Relevance | 3.92 / 5 | 4.67 / 5 | 4.50 / 5 |
| Completeness | 2.58 / 5 | 3.33 / 5 | 3.70 / 5 |
| Redundancy | 1.96 / 5 | 1.33 / 5 | 1.30 / 5 |

## Interpretation

V3 is the best current pilot result. It confirms that candidate evidence
retrieval is necessary before LLM semantic segmentation when the WildClawBench
workspace contains many unrelated files or outdated local references.

The next report should present V3 as the main result and mention V1/V2 as
ablation steps.
