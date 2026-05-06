# OpenClaw + SGLang Prefix Cache Baseline

This repository contains the initial setup notes and benchmark artifacts for testing SGLang RadixAttention prefix cache behavior before implementing sub-context-aware KV cache reuse.

## Contents

- `bench_sglang_prefix_cache.py`  
  Sends R1/R2/R3 prompts to SGLang and exports CSV/JSON benchmark results.

- `benchmark_results/`  
  Captured baseline CSV/JSON outputs.

- `docs/baseline_report.md`  
  Short report explaining the prefix-cache baseline result.

- `docs/sglang_source_map.md`  
  Source map for the SGLang files involved in radix prefix-cache lookup and KV/cache logging.

- `docs/neno5_setup.md`  
  Step-by-step instructions for reproducing the initial OpenClaw + SGLang runtime setup on `neno5`/`nano5`.

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

