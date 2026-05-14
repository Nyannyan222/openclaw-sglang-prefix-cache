# WildClaw Similarity Task Selector

- Created at: `2026-05-14T18:04:16+00:00`
- WildClaw root: `C:/Users/Administrator/OneDrive/文件/New project/external/WildClawBench`
- Tasks scanned: 60
- Selected top tasks: 1
- Minimum selected score: 1.0

## Top Candidates

| rank | score | task | category | files | segments | exact dup groups | near dup pairs | top sim | hints |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 73.0 | `04_Search_Retrieval_task_2_conflicting_handling` | `04_Search_Retrieval` | 14 | 148 | 0 | 0 | 0.0 | api_docs, law_policy, safety, instructions |

## Why These Tasks

### 1. `04_Search_Retrieval_task_2_conflicting_handling`

- Score: `73.0`
- Reasons: workspace available; 14 included workspace files; domain hints: api_docs, law_policy, safety, instructions; long/constraint-heavy prompt

## Suggested Next Commands

Use the selected tasks as the next real-context extraction input:

```powershell
.\.venv\Scripts\python.exe scripts\real_context_extractor.py `
  --wildclaw-root external\WildClawBench `
  --output-dir benchmark_results\wildclaw_similarity_selected_pilot `
  --task 04_Search_Retrieval_task_2_conflicting_handling
```

Then run semantic segmentation and strict similarity discovery on that selected pilot.
