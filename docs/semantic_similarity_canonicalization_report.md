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

## Local RTX 5070 Expanded Result

The expanded 5-pair manifest was replayed locally through SGLang Docker on the
RTX 5070 using `Qwen/Qwen2.5-0.5B-Instruct`.

Source:

- `benchmark_results/wildclaw_sglang_runtime_runs/wildclaw_sglang_runtime_20260514_202320/wildclaw_sglang_runtime_results.csv`
- `benchmark_results/wildclaw_sglang_runtime_runs/wildclaw_sglang_runtime_20260514_202320/semantic_similarity_runtime_pair_summary.md`

Aggregate result:

| condition | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `canonical_context` | 5 | 189 | 73.6 | 115.4 | 38.9% | 0.7757 |
| `similar_context` | 5 | 163 | 77.8 | 85.2 | 47.7% | 0.6867 |
| `canonical_plus_delta` | 5 | 305 | 183 | 122 | 60.0% | 0.6545 |
| `canonical_context_repeat` | 5 | 190 | 162 | 28 | 85.3% | 0.6322 |

Per-pair result:

| pair | similarity | delta chars | similar ratio | canonical+delta ratio | gain | similar prefill | canonical+delta prefill | latency delta s | repeat ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sim_pair_001` | 0.1836 | 275 | 18.8% | 66.5% | 47.7% | 121 | 108 | 0.0094 | 87.3% |
| `sim_pair_002` | 0.1784 | 504 | 85.0% | 52.6% | -32.4% | 33 | 179 | 0.0361 | 86.4% |
| `sim_pair_003` | 0.1192 | 75 | 21.1% | 61.3% | 40.2% | 112 | 101 | -0.2143 | 83.2% |
| `sim_pair_004` | 0.1052 | 275 | 77.9% | 64.8% | -13.0% | 33 | 108 | 0.0228 | 86.4% |
| `sim_pair_005` | 0.0566 | 254 | 18.1% | 55.6% | 37.6% | 127 | 114 | -0.0151 | 81.3% |

Summary:

- `canonical_plus_delta` improves cached-token ratio over native
  `similar_context` in 3 of 5 pairs.
- Mean cached-ratio gain is 16.0 percentage points.
- Mean latency delta is -0.0322 seconds in this local run.
- `canonical_context_repeat` remains the upper bound, reaching about 85.3%
  cached-token ratio.

The two negative pairs are not failures of exact-prefix canonicalization itself.
They show that the sequential replay order can let `similar_context` benefit
from prefix cache entries created by earlier pairs. This motivates an isolated
or randomized replay protocol where each pair is evaluated after a clean cache
state.

Run locally:

```powershell
.\scripts\start_local_sglang_docker.ps1
.\scripts\run_semantic_similarity_local_or_neno5.ps1 -RunRuntime -Limit 0 -Repeat 1
.\.venv\Scripts\python.exe scripts\summarize_semantic_similarity_runtime.py `
  benchmark_results\wildclaw_sglang_runtime_runs\<run_dir>\wildclaw_sglang_runtime_results.csv
```

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
research step is to run isolated or randomized per-pair replay to remove cache
order effects, then create a richer similarity-pair dataset from WildClaw tasks
with repeated or near-duplicate evidence, such as legal/versioned documents, API
documentation, README variants, issue/PR duplicates, and safety prompt injection
variants.
