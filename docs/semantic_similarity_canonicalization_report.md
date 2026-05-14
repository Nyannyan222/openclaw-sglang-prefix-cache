# Semantic Sub-context Canonicalization Report

## Method

We propose **Semantic Sub-context Canonicalization with Delta Preservation**.
The method does not directly reuse KV tensors for token-different semantic
paraphrases. Instead, it converts semantic/content similarity into exact-token
cache reuse before runtime.

Pipeline:

1. Extract WildClawBench real task context from task prompts and available workspace files.
2. Segment long context into semantic sub-contexts.
3. Detect similar sub-context pairs using normalized content similarity.
4. Select one reusable canonical context per similar pair or group.
5. Preserve task-specific differences as a delta.
6. Build stable prompts where `canonical_plus_delta` starts with the exact same prefix as `canonical_context`.
7. Replay through SGLang and compare cached tokens, prefill tokens, and latency.

## neno5 Result

Source: SGLang runtime replay on neno5 using the semantic similarity protocol.

| condition | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `canonical_context` | 220 |  |  |  | 0.5928 |
| `similar_context` | 149 | 28 | 121 | 18.8% | 0.5926 |
| `canonical_plus_delta` | 322 | 214 | 108 | 66.5% | 0.9203 |
| `canonical_context_repeat` | 221 | 193 | 28 | 87.3% | 0.5815 |

## Interpretation

`similar_context` is the native baseline: the context is content-similar but
token-different, so SGLang only reuses a small fixed prefix.

`canonical_plus_delta` is the proposed method. It starts with the exact canonical
prefix and appends a task-specific delta. This raises cached-token ratio from
18.8% to 66.5%, showing that semantic/content similarity can be converted into
runtime cache reuse through canonicalization.

`canonical_context_repeat` is the upper-bound exact-repeat case. It reaches
87.3% cached-token ratio. `canonical_plus_delta` is lower because the delta still
requires prefill, but it is substantially better than native similar-context
reuse.

## Expanded Pair Set

The current expanded manifest uses the V3 WildClaw Search Retrieval semantic
sub-contexts and lowers the similarity threshold to `0.05`, producing 5
candidate pairs and 20 runtime replay rows:

- `canonical_context`
- `similar_context`
- `canonical_plus_delta`
- `canonical_context_repeat`

Artifacts:

- `benchmark_results/semantic_similarity_kv_reuse/semantic_similarity_pairs.csv`
- `benchmark_results/semantic_similarity_kv_reuse/semantic_similarity_runtime_manifest.jsonl`
- `benchmark_results/semantic_similarity_kv_reuse/semantic_similarity_block_manifest.jsonl`

Run on neno5:

```bash
git pull origin main

sbatch --account=MST114180 \
  --export=ALL,MANIFEST=benchmark_results/semantic_similarity_kv_reuse/semantic_similarity_runtime_manifest.jsonl,LIMIT=0,REPEAT=1 \
  scripts/slurm_run_wildclaw_runtime_replay.sh
```

## Current Limitation

This is still a small proof-of-concept. The V3 pilot has only 10 semantic
sub-contexts, so the expanded manifest contains 5 candidate pairs. The next
research step is to create a richer similarity-pair dataset from WildClaw tasks
with repeated or near-duplicate evidence, such as legal/versioned documents,
API documentation, README variants, issue/PR duplicates, and safety prompt
injection variants.
