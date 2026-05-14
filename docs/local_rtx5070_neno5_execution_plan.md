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

Current local status:

- WSL2 Ubuntu is installed.
- WSL2 can see the RTX 5070 through `nvidia-smi`.
- Docker Desktop is installed and Docker GPU pass-through works.
- The SGLang Docker runtime can serve `Qwen/Qwen2.5-0.5B-Instruct` locally on
  `http://127.0.0.1:30000/v1`.

The native Windows project `.venv` still does not need PyTorch/SGLang. Runtime
serving is handled by Docker, while the replay and report scripts run from the
local repository.

## Start Local SGLang Docker

```powershell
.\scripts\start_local_sglang_docker.ps1
```

This starts:

- Container: `openclaw-sglang-local`
- Image: `lmsysorg/sglang:latest-runtime`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Endpoint: `http://127.0.0.1:30000/v1`

The script installs the missing `distro` Python package inside the container
before launching SGLang. This is needed for the current `latest-runtime` image
because the OpenAI package imports `distro` during server startup.

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

For the cleaner isolated pair protocol, flush the SGLang cache before each
similarity pair:

```powershell
.\scripts\run_semantic_similarity_local_or_neno5.ps1 -RunRuntime -Limit 0 -Repeat 1 -FlushBefore pair
```

Local validation on 2026-05-14 succeeded with 20 rows and 0 errors.

| condition | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `canonical_context` | 5 | 189 | 73.6 | 115.4 | 38.9% | 0.7757 |
| `similar_context` | 5 | 163 | 77.8 | 85.2 | 47.7% | 0.6867 |
| `canonical_plus_delta` | 5 | 305 | 183 | 122 | 60.0% | 0.6545 |
| `canonical_context_repeat` | 5 | 190 | 162 | 28 | 85.3% | 0.6322 |

The isolated pair replay on 2026-05-14 also succeeded with 20 rows and 0
errors. Because it flushes cache before each pair, it removes cross-pair order
effects:

| condition | rows | avg prompt | avg cached | avg prefill | cached ratio | avg latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `canonical_context` | 5 | 189 | 0 | 189 | 0.0% | 0.7134 |
| `similar_context` | 5 | 163 | 28.4 | 134.6 | 17.4% | 0.6608 |
| `canonical_plus_delta` | 5 | 305 | 183 | 122 | 60.0% | 0.6836 |
| `canonical_context_repeat` | 5 | 190 | 162 | 28 | 85.3% | 0.6733 |

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
