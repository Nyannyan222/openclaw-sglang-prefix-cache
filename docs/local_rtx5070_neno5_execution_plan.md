# Local RTX 5070 / neno5 Execution Plan

## Decision Rule

Use the local RTX 5070 machine for work that does not require a cluster SGLang
runtime:

- WildClawBench data inspection and leakage checks.
- Real-context extraction from task prompts and workspace files.
- Semantic segmentation when using API-backed LLM calls.
- Fixed-size, file-based, and semantic sub-context JSONL generation.
- Semantic/content similarity grouping.
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

## Semantic Discovery

Run strict semantic sub-context discovery locally:

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

This produces:

- `benchmark_results/semantic_similarity_discovery_openai_strict/semantic_similar_subcontext_pairs.csv`
- `benchmark_results/semantic_similarity_discovery_openai_strict/semantic_similar_subcontext_pairs.jsonl`
- `benchmark_results/semantic_similarity_discovery_openai_strict/semantic_similarity_manual_review.csv`

The current strict V3 pilot result is:

| metric | value |
| --- | ---: |
| candidate pairs evaluated | 45 |
| semantic matches | 0 |
| semantic groups | 0 |

## neno5 Runtime Replay

Use neno5 for larger SGLang runtime replays or larger models. After
pushing/pulling the latest repo on neno5:

```bash
git pull origin main

sbatch --account=MST114180 scripts/slurm_run_wildclaw_runtime_replay.sh
```

View the latest summary:

```bash
RUN_DIR=$(ls -td benchmark_results/wildclaw_sglang_runtime_runs/wildclaw_sglang_runtime_* | head -1)
cat "$RUN_DIR/wildclaw_sglang_runtime_summary.md"
```

## What Counts As Success

For the semantic discovery stage, success means producing high-quality
sub-context pairs or groups whose evidence role is genuinely similar under
embedding, LLM judge, and manual review. Related-but-different sub-contexts
should be rejected or marked for review rather than counted as matches.
