# Manual Review Report V2

Reviewed file:

```text
wildclaw_semantic_subcontext_pilot_v2/wildclaw_semantic_subcontext_pilot.jsonl
```

Reviewed annotations:

```text
wildclaw_manual_review_sheet_v2_reviewed_by_codex.csv
```

## Review Summary

| Status | Count |
| --- | ---: |
| Pass | 5 |
| Revise | 4 |
| Reject | 0 |
| Total | 9 |

Average manual scores:

| Metric | V1 Average | V2 Average |
| --- | ---: | ---: |
| Independence | 4.71 / 5 | 4.67 / 5 |
| Relevance | 3.92 / 5 | 4.67 / 5 |
| Completeness | 2.58 / 5 | 3.33 / 5 |
| Redundancy | 1.96 / 5 | 1.33 / 5 |

## Main Finding

V2 improves quality over V1. The LLM no longer produces task-id, category, or
task-name-only fragments, and the sub-contexts are more consolidated and
evidence-like. Relevance and completeness both improve, while redundancy drops.

However, V2 is still not the final evidence-bearing dataset. For
`04_Search_Retrieval_task_2_conflicting_handling`, the task requires resolving
conflicts between local legal materials and web evidence. V2 captures the case
facts and identifies the legal-materials folder, but it does not extract actual
statutory snippets from the local law files. That means the key conflict
resolution evidence is still missing.

## V1 vs V2

| Version | Rows | Main Strength | Main Weakness |
| --- | ---: | --- | --- |
| V1 | 24 | Enough rows for manual review; high independence. | Too many short metadata/constraint fragments. |
| V2 | 9 | Richer, less redundant, better relevance and completeness. | Too few rows and still lacks law-file evidence for task 2. |

## Recommendation For V3

Create a targeted evidence extraction stage before LLM segmentation:

1. For each workspace file, extract candidate evidence passages using keywords
   from the task prompt.
2. For task 2, search local legal materials for terms such as statute of
   limitations, debt, acknowledgment, claim period, interruption, suspension,
   and limitation period.
3. Pass only these candidate passages plus the task prompt into the LLM.
4. Ask the LLM to produce 12-18 evidence-bearing sub-contexts across the three
   pilot tasks.

The V3 prompt should explicitly require:

```text
Each sub-context must include at least one quoted or near-verbatim evidence
snippet from PROMPT or a workspace file. Do not cite a source file unless the
sub-context includes actual evidence from that file.
```

Expected result:

```text
V3 should preserve V2's higher relevance while improving completeness for
local-file evidence and conflict-resolution tasks.
```
