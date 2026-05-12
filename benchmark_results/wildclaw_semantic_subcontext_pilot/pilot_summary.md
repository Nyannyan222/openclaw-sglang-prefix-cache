# WildClawBench Semantic Sub-Context Pilot Summary

Generated from `04_Search_Retrieval` pilot tasks:

| Task | Included Files | Skipped Files | Long Context Chars |
| --- | ---: | ---: | ---: |
| `04_Search_Retrieval_task_2_conflicting_handling` | 14 | 2 | 279749 |
| `04_Search_Retrieval_task_4_efficient_search` | 1 | 0 | 1548 |
| `04_Search_Retrieval_task_5_fuzzy_search` | 1 | 0 | 514 |

Current segmentation run:

| Method | Sub-Context Count | Avg Context Reduction | Avg Independence | Avg Relevance | Avg Completeness | Avg Redundancy Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| File-based chunking | 17 | 0.8549 | 0.7412 | 0.1765 | 0.8667 | 0.3618 |
| Fixed-size chunking | 80 | 0.9611 | 0.6000 | 0.0724 | 0.6721 | 0.3500 |
| LLM semantic sub-context | 24 | 0.9337 | 0.9417 | 0.9958 | 0.9375 | 0.0417 |

Notes:

- `wildclaw_semantic_subcontext_pilot.jsonl` currently contains 24 LLM-generated
  rows, enough for the planned 20-30 row manual inspection.
- The latest successful run used `gpt-4o-mini` through the OpenAI-compatible
  chat completions endpoint. The run log reports 8 semantic sub-contexts for
  each of the three selected Search Retrieval tasks.
- Unsupported PDFs are retained as metadata in the extracted long context and
  excluded from text segmentation until a PDF text extractor is added.
