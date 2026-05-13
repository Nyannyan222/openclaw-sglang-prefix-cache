# WildClaw Mixed-Category Evaluation

## Scope

- Categories: `01_Productivity_Flow`, `04_Search_Retrieval`, `06_Safety_Alignment`
- Manifest rows: 21
- Conditions: `fixed_size_chunks`, `file_based_chunks`, `semantic_subcontext`

## Category Summary

| category | condition | tasks | missing workspaces | avg context chars | avg reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| `01_Productivity_Flow` | `file_based_chunks` | 2 | 2 | 3195.5 | 0.0021 |
| `01_Productivity_Flow` | `fixed_size_chunks` | 2 | 2 | 3350.0 | -0.0478 |
| `01_Productivity_Flow` | `semantic_subcontext` | 2 | 2 | 3149.5 | 0.0167 |
| `04_Search_Retrieval` | `file_based_chunks` | 3 | 0 | 100030.3 | 0.0535 |
| `04_Search_Retrieval` | `fixed_size_chunks` | 3 | 0 | 114946.0 | -0.1775 |
| `04_Search_Retrieval` | `semantic_subcontext` | 3 | 0 | 11013.7 | 0.3386 |
| `06_Safety_Alignment` | `file_based_chunks` | 2 | 2 | 368.5 | -0.0249 |
| `06_Safety_Alignment` | `fixed_size_chunks` | 2 | 2 | 506.5 | -0.4081 |
| `06_Safety_Alignment` | `semantic_subcontext` | 2 | 2 | 392.5 | -0.0921 |

## Notes

- Search Retrieval rows use the available local WildClawBench workspace files.
- Productivity and Safety rows are prompt-only if their workspace directories are absent in the local checkout.
- The same manifest shape can be passed to `scripts/run_wildclaw_framework_eval.py`.
