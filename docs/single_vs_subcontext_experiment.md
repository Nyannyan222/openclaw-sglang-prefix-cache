# Single-Context vs Sub-Context Cache Experiment

This experiment compares three cache conditions over the same request sequence:

| Group | Description |
| --- | --- |
| Baseline 1: No cache | Start SGLang with `--disable-radix-cache`. |
| Baseline 2: SGLang RadixAttention | Start SGLang with its native RadixAttention prefix cache. |
| Proposed: Sub-context-aware cache | Run the metadata-level `SubContextIndex` prototype outside SGLang. |

## Request Sequence

```text
R1 = A + B + C
R2 = A + B + C
R3 = C + A + B
R4 = B + C + A
R5 = A + C + B
```

All requests reuse the same A/B/C content but change the order after R2. This
keeps the experiment focused on the difference between ordered prefix reuse and
segment-level reuse.

## Metrics

| Metric | Meaning |
| --- | --- |
| cached token ratio | `cached_tokens / prompt_tokens`. |
| prefill tokens | Estimated tokens that still require prefill: `prompt_tokens - cached_tokens`. |
| first token latency | For the non-streaming benchmark, `prefill_finished_ts - request_received_ts` from SGLang logs is used as a TTFT proxy. |
| total latency | SGLang `e2e_latency` when present, otherwise client-side elapsed time. |

For the proposed prototype, token metrics are estimated from metadata. Latency
is marked `not_measured_metadata_prototype` until the runtime can actually
reuse non-prefix KV spans.

## WSL Command

```bash
bash scripts/wsl_run_cache_baseline_matrix.sh
```

The matrix script runs:

1. `CACHE_MODE=no_cache bash scripts/wsl_run_qwen05_benchmark.sh`
2. `CACHE_MODE=radix bash scripts/wsl_run_qwen05_benchmark.sh`
3. `python scripts/subcontext_cache_prototype.py <radix-json>`
4. `python scripts/compare_cache_baselines.py ...`

The combined comparison output is:

```text
benchmark_results/wsl_cache_baseline_matrix_<timestamp>/
  single_vs_subcontext_cache_comparison_<timestamp>.csv
  single_vs_subcontext_cache_comparison_<timestamp>.json
```

## Expected Pattern

```text
R1: all groups start cold.
R2: RadixAttention should reuse most of A+B+C because the prefix is repeated.
R3/R4/R5: RadixAttention should drop because the prefix order changes.
R3/R4/R5: SubContextIndex should still identify A, B, and C by content hash.
```

The safe conclusion to report is:

```text
The prototype demonstrates additional segment-level reuse opportunities beyond
native prefix reuse. It does not yet measure latency improvement because real
non-prefix KV reuse needs runtime support.
```
