# OpenClaw + SGLang Prefix Cache Baseline

This repository contains the initial setup notes and benchmark artifacts for testing SGLang RadixAttention prefix cache behavior before implementing sub-context-aware KV cache reuse.

## Contents

- `bench_sglang_prefix_cache.py`  
  Sends R1/R2/R3 prompts to SGLang and exports request-level CSV/JSON results plus sub-context metadata for A/B/C spans.

- `benchmark_results/`  
  Captured baseline CSV/JSON outputs.

- `docs/baseline_report.md`  
  Short report explaining the prefix-cache baseline result.

- `docs/sglang_source_map.md`  
  Source map for the SGLang files involved in radix prefix-cache lookup and KV/cache logging.

- `docs/neno5_setup.md`  
  Step-by-step instructions for reproducing the initial OpenClaw + SGLang runtime setup on `neno5`/`nano5`.

- `scripts/slurm_setup_and_benchmark.sh`  
  SLURM job that installs OpenClaw/SGLang on a compute node, starts SGLang with logging/metrics, configures OpenClaw, and runs the benchmark.

- `scripts/slurm_setup_env.sh`
  One-time setup job. It installs Node.js, OpenClaw, uv, and the pinned SGLang runtime under `/work/$USER/openclaw-sglang`.

- `scripts/slurm_run_benchmark.sh`
  Benchmark-only job. It reuses the setup runtime, starts SGLang, and runs R1/R2/R3.

- `scripts/slurm_run_model_matrix.sh`
  Submits one benchmark job per model for comparing prefix-cache behavior across models.

- `scripts/neno5_login_node_check.sh`  
  Lightweight login-node sanity check. It does not run SGLang or GPU workload.

## Baseline Result

Latest benchmark:

```text
benchmark_results/sglang_prefix_cache_20260506_131529.csv
```

Observed cached-token counts:

```text
R1 = A+B+C -> cached_tokens 24
R2 = A+B+C -> cached_tokens 354
R3 = C+A+B -> cached_tokens 71
```

Interpretation:

SGLang's current radix prefix cache reuses KV cache well for repeated ordered prefixes, but reuse drops when the same sub-contexts are reordered. This motivates the next step: sub-context-aware cache instrumentation and design.

## Quick Benchmark Command

With SGLang running on `http://127.0.0.1:30000/v1`:

```bash
python3 bench_sglang_prefix_cache.py \
  --output-dir benchmark_results \
  --timeout 180 \
  --max-tokens 64
```

Each run writes three artifacts:

```text
sglang_prefix_cache_<timestamp>.csv
sglang_prefix_cache_<timestamp>.json
sglang_prefix_cache_<timestamp>_subcontexts.csv
```

The `_subcontexts.csv` file records `request_id`, `subcontext_id`,
`char_start`, `char_end`, `token_start`, `token_end`, `token_len`,
`content_hash`, `order`, `cached_tokens`, `prompt_tokens`, and `cache_ratio`.

The SLURM workflow also patches SGLang with cache lookup logging. When
`SGLANG_PREFIX_CACHE_DEBUG_LOG=1` is set, the SGLang log includes JSON lines
with:

```text
event=cache_lookup
rid
input_token_len
matched_prefix_len
matched_node_id
cached_tokens
uncached_tokens
first_mismatch_token_position
```

The main benchmark CSV/JSON copies those fields into columns prefixed with
`lookup_`, for example `lookup_matched_prefix_len` and
`lookup_first_mismatch_token_position`.

## neno5/nano5 Quick Start

On the login node:

```bash
git clone https://github.com/Nyannyan222/openclaw-sglang-prefix-cache.git
cd openclaw-sglang-prefix-cache
bash scripts/neno5_login_node_check.sh
sbatch --account=MST114180 scripts/slurm_setup_env.sh
sbatch --account=MST114180 scripts/slurm_run_benchmark.sh
```

Run a single alternate model:

```bash
sbatch --account=MST114180 --export=ALL,MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct scripts/slurm_run_benchmark.sh
```

Run the default model matrix:

```bash
bash scripts/slurm_run_model_matrix.sh
```

Customize the model matrix:

```bash
MODELS="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct" \
bash scripts/slurm_run_model_matrix.sh
```

Do not start SGLang directly on the login node. The SLURM script runs it inside
a GPU job and installs a local Node.js runtime under `/work/$USER/openclaw-sglang/node`
if the cluster does not provide `node`/`npm`.
