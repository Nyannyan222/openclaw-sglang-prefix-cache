# Local RTX 5070 / neno5 Execution Plan

## Decision Rule

Use the local RTX 5070 machine for work that does not require a cluster SGLang
runtime:

- WildClawBench data inspection and leakage checks.
- Real-context extraction from task prompts and workspace files.
- Semantic segmentation when using API-backed LLM calls.
- Fixed-size, file-based, and semantic sub-context JSONL generation.
- Semantic/content similarity grouping.
- Canonical context plus delta prompt construction.
- Report generation from CSV/JSON artifacts.

Use neno5 for SGLang runtime experiments unless a local SGLang endpoint is
already running:

- SGLang server startup.
- Runtime replay against `/v1/chat/completions`.
- Cached-token, prefill-token, and latency measurements.
- Larger model or larger manifest runs.

Current local status: the machine has an RTX 5070 with about 12 GB VRAM, but
the project `.venv` does not currently include PyTorch/SGLang. Therefore the
local machine is ready for manifest/report stages, while runtime replay should
go to neno5 unless a local SGLang runtime is installed separately.

## One-Command Local Preparation

Prepare the expanded semantic-similarity manifest locally:

```powershell
.\scripts\run_semantic_similarity_local_or_neno5.ps1
```

This regenerates:

- `benchmark_results/semantic_similarity_kv_reuse/semantic_similarity_pairs.csv`
- `benchmark_results/semantic_similarity_kv_reuse/semantic_similarity_runtime_manifest.jsonl`
- `benchmark_results/semantic_similarity_kv_reuse/semantic_similarity_block_manifest.jsonl`
- `benchmark_results/semantic_similarity_kv_reuse/semantic_similarity_protocol.json`

If a local SGLang endpoint is already running on `http://127.0.0.1:30000/v1`,
run the runtime replay locally:

```powershell
.\scripts\run_semantic_similarity_local_or_neno5.ps1 -RunRuntime
```

If the endpoint is absent, the script prints the matching neno5 `sbatch`
command instead of failing.

## neno5 Runtime Replay

After pushing/pulling the latest repo on neno5:

```bash
git pull origin main

sbatch --account=MST114180 \
  --export=ALL,MANIFEST=benchmark_results/semantic_similarity_kv_reuse/semantic_similarity_runtime_manifest.jsonl,LIMIT=0,REPEAT=1 \
  scripts/slurm_run_wildclaw_runtime_replay.sh
```

View the latest summary:

```bash
RUN_DIR=$(ls -td benchmark_results/wildclaw_sglang_runtime_runs/wildclaw_sglang_runtime_* | head -1)
cat "$RUN_DIR/wildclaw_sglang_runtime_summary.md"
```

## What Counts As Success

For the expanded similarity-pair run, compare `similar_context` with
`canonical_plus_delta`:

- `canonical_plus_delta` should have a higher cached-token ratio.
- `canonical_plus_delta` should show fewer uncached/prefill tokens relative to
  its reusable canonical prefix.
- `canonical_context_repeat` remains the exact-repeat upper bound.

The important claim is not that token-different semantic paraphrases can be
directly spliced into KV cache. The claim is that semantic/content similarity
can identify reusable evidence, then canonicalization turns it into exact-token
prefix reuse while the delta preserves task-specific details.
