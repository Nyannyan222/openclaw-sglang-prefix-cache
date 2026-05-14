# Semantic Sub-context Similarity Discovery

## Direction Change

We stop using `canonical_plus_delta` as the main method.

The new objective is to find genuinely semantically similar WildClawBench
sub-contexts. The output should be semantic pairs or semantic groups, not prompt
prefix rewrites for SGLang cache reuse.

## What This Means

Previous direction:

```text
semantic/content candidate
→ canonical prefix + delta
→ exact-prefix reuse in SGLang
```

New direction:

```text
WildClaw semantic sub-contexts
→ lexical prefilter
→ embedding similarity
→ optional LLM judge
→ semantic similar pairs/groups
→ manual review
```

The lexical score is no longer treated as the final similarity signal. It is
only a cheap candidate generator.

## Similarity Criteria

A pair should be considered semantically similar only if it satisfies one of
these stronger conditions:

- High embedding cosine similarity.
- LLM judge score at least 3 out of 4.
- Manual review confirms that the two sub-contexts provide the same or highly
  overlapping evidence role for a task.

The current judge rubric is:

| score | meaning |
| ---: | --- |
| 0 | unrelated or contradictory |
| 1 | same broad topic only |
| 2 | partially overlapping evidence or role |
| 3 | highly similar meaning but different details |
| 4 | near-equivalent meaning/evidence |

## New Script

```powershell
.\.venv\Scripts\python.exe scripts\find_semantic_similar_subcontexts.py `
  --semantic-jsonl benchmark_results\wildclaw_semantic_subcontext_pilot_v3\wildclaw_semantic_subcontext_pilot.jsonl `
  --output-dir benchmark_results\semantic_similarity_discovery `
  --backend openai_embedding_judge `
  --prefilter-threshold 0.02 `
  --embedding-threshold 0.72 `
  --judge-threshold 3 `
  --max-candidates 20
```

If `OPENAI_API_KEY` is not set, use the prefilter-only mode to produce a manual
review queue:

```powershell
.\.venv\Scripts\python.exe scripts\find_semantic_similar_subcontexts.py `
  --semantic-jsonl benchmark_results\wildclaw_semantic_subcontext_pilot_v3\wildclaw_semantic_subcontext_pilot.jsonl `
  --output-dir benchmark_results\semantic_similarity_discovery `
  --backend lexical_prefilter `
  --prefilter-threshold 0.02 `
  --max-candidates 20
```

## Current Local Output

The earlier prefilter-only run produced a manual review queue:

- Output directory: `benchmark_results/semantic_similarity_discovery`
- Candidates: 6
- Semantic matches: 0
- Groups: 0

The zero matches do not mean there are no semantic matches. It means the run did
not use embeddings or LLM judge, so each candidate is marked
`needs_semantic_judge`.

After creating a local `OPENAI_API_KEY`, the stricter OpenAI-backed discovery
run evaluated all 45 same-category pairs from the V3 pilot:

```powershell
.\.venv\Scripts\python.exe scripts\find_semantic_similar_subcontexts.py `
  --semantic-jsonl benchmark_results\wildclaw_semantic_subcontext_pilot_v3\wildclaw_semantic_subcontext_pilot.jsonl `
  --output-dir benchmark_results\semantic_similarity_discovery_openai_strict `
  --backend openai_embedding_judge `
  --prefilter-threshold 0 `
  --embedding-threshold 0.72 `
  --judge-threshold 3 `
  --min-match-embedding 0.50 `
  --require-same-answer-utility `
  --max-candidates 45 `
  --same-category-only
```

Strict OpenAI-backed result:

| metric | value |
| --- | ---: |
| candidate pairs evaluated | 45 |
| semantic matches | 0 |
| semantic groups | 0 |
| embedding model | `text-embedding-3-small` |
| judge model | `gpt-4o-mini` |

This is an important finding: the current V3 WildClaw Search Retrieval pilot
does contain related sub-contexts, but under a strict semantic-similarity
definition they are not near-equivalent or interchangeable. The high-scoring
examples are partial-overlap pairs, such as statute-of-limitations duration vs
statute-of-limitations suspension, or task requirements vs output format. These
belong to the same task or domain, but they provide different evidence roles.

Artifacts:

- `benchmark_results/semantic_similarity_discovery/semantic_similar_subcontext_pairs.csv`
- `benchmark_results/semantic_similarity_discovery/semantic_similar_subcontext_pairs.jsonl`
- `benchmark_results/semantic_similarity_discovery/semantic_similar_subcontext_groups.jsonl`
- `benchmark_results/semantic_similarity_discovery/semantic_similarity_manual_review.csv`
- `benchmark_results/semantic_similarity_discovery/semantic_similarity_discovery_protocol.json`

OpenAI-backed strict artifacts:

- `benchmark_results/semantic_similarity_discovery_openai_strict/semantic_similar_subcontext_pairs.csv`
- `benchmark_results/semantic_similarity_discovery_openai_strict/semantic_similar_subcontext_pairs.jsonl`
- `benchmark_results/semantic_similarity_discovery_openai_strict/semantic_similar_subcontext_groups.jsonl`
- `benchmark_results/semantic_similarity_discovery_openai_strict/semantic_similarity_manual_review.csv`
- `benchmark_results/semantic_similarity_discovery_openai_strict/semantic_similarity_discovery_protocol.json`

## Next Experiment

The next experiment should expand beyond the small V3 Search Retrieval pilot.
Use more WildClaw tasks and categories, then rerun the strict
`openai_embedding_judge` pipeline. The deliverable should be a semantic
similarity dataset:

```text
wildclaw_semantic_similar_subcontexts.jsonl
```

Each row should contain:

- `pair_id`
- `left_id`
- `right_id`
- `embedding_cosine`
- `llm_judge_score`
- `semantic_relation`
- `manual_label`
- `semantic_decision`
- `rationale`

This dataset can later be used for retrieval, clustering, redundancy analysis,
or a separate runtime cache design. It should not assume `canonical_plus_delta`.
