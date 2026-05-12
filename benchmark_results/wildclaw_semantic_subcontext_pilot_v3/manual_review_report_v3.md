# Manual Review Report V3

Reviewed file:

```text
wildclaw_semantic_subcontext_pilot_v3/wildclaw_semantic_subcontext_pilot.jsonl
```

Reviewed annotations:

```text
wildclaw_manual_review_sheet_v3_reviewed_by_codex.csv
```

## Review Summary

| Status | Count |
| --- | ---: |
| Pass | 8 |
| Revise | 1 |
| Reject | 1 |
| Total | 10 |

Average manual scores:

| Metric | V1 Avg | V2 Avg | V3 Avg |
| --- | ---: | ---: | ---: |
| Independence | 4.71 / 5 | 4.67 / 5 | 4.80 / 5 |
| Relevance | 3.92 / 5 | 4.67 / 5 | 4.50 / 5 |
| Completeness | 2.58 / 5 | 3.33 / 5 | 3.70 / 5 |
| Redundancy | 1.96 / 5 | 1.33 / 5 | 1.30 / 5 |

## Main Finding

V3 is the strongest pilot version so far. It adds a candidate evidence retrieval
stage before LLM segmentation, which fixes the main V2 weakness: task 2 now
contains real local-law evidence from `law12.pdf` instead of only case facts or
folder references.

The most important extracted evidence is:

```text
第一百三十五条　向人民法院请求保护民事权利的诉讼时效期间为二年，法律另有规定的除外。
```

This is useful because the task expects the framework to detect that local legal
materials may be outdated and then verify against current web evidence. The
local evidence therefore supplies the conflict side of the conflict-resolution
evaluation.

## Version Comparison

| Version | Rows | Strength | Weakness |
| --- | ---: | --- | --- |
| V1 | 24 | Enough manual-review rows. | Too many short metadata fragments. |
| V2 | 9 | Better consolidation and fewer metadata fragments. | Still lacks actual local-law evidence. |
| V3 | 10 | Includes candidate local-law evidence and better completeness. | Fewer rows; one weak civil/criminal responsibility passage should be removed. |

## Recommended Dataset Use

Use V3 as the main pilot dataset, but filter out rows with
`manual_review_status = reject`.

Suggested final pilot set:

```text
8 pass rows
1 revise row kept only if the report needs broader legal background
1 reject row excluded
```

The report claim should be:

```text
Adding candidate evidence retrieval before LLM semantic segmentation improves
completeness and makes the sub-contexts more suitable for real-context
evaluation than direct LLM segmentation over the whole long context.
```
