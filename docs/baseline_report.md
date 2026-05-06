# SGLang Radix Prefix Cache Baseline

## Goal

This baseline verifies the current SGLang RadixAttention prefix-cache behavior before implementing sub-context-aware KV reuse.

The experiment checks whether SGLang can reuse KV cache when:

- The second request shares the same ordered context prefix as the first request.
- The third request contains the same sub-contexts but in a different order.

## Runtime Setup

- OpenClaw CLI: `OpenClaw 2026.5.4`
- SGLang: `0.5.9`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Endpoint: `http://127.0.0.1:30000/v1`
- Metrics endpoint: `http://127.0.0.1:30000/metrics`
- SGLang log file: `/tmp/sglang_openclaw.log`

SGLang was launched with request logging and metrics enabled:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-metrics \
  --log-requests \
  --log-requests-level 1 \
  --log-requests-format json \
  --radix-eviction-policy lru \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph
```

RadixAttention prefix caching is enabled because `disable_radix_cache=False` in the server log. The `--disable-radix-cache` flag was not used.

The Triton/PyTorch backend is used for this baseline because the local RTX 5070 environment currently fails FlashInfer JIT compilation with `compute_120a` under the available `nvcc` path. This does not disable SGLang's radix prefix-cache layer.

## Benchmark Script

Script:

```text
bench_sglang_prefix_cache.py
```

Latest output files:

```text
benchmark_results/sglang_prefix_cache_20260506_131529.csv
benchmark_results/sglang_prefix_cache_20260506_131529.json
```

The script sends three requests:

```text
R1 = A + B + C + question1
R2 = A + B + C + question2
R3 = C + A + B + question3
```

`A`, `B`, and `C` are semantically independent product sub-contexts.

## Results

| Request | Sub-context order | Prompt tokens | Cached tokens | Interpretation |
|---|---:|---:|---:|---|
| R1_ABC | A+B+C | 366 | 24 | Mostly cold request, except small shared benchmark boilerplate. |
| R2_ABC_same_prefix | A+B+C | 367 | 354 | Strong prefix-cache reuse because the ordered prefix matches R1. |
| R3_CAB_reordered | C+A+B | 367 | 71 | Much lower reuse even though A, B, and C all appeared before. |

## Observations

SGLang's current radix prefix cache works well when the new request begins with the same token prefix as an earlier request.

The second request, `R2_ABC_same_prefix`, reuses `354` cached tokens, nearly the whole prompt prefix.

The third request, `R3_CAB_reordered`, contains the same sub-contexts but changes their order. It only reuses `71` cached tokens. This shows that the existing mechanism is prefix-sensitive: reusable content in a different position is not fully reused.

## Conclusion

This baseline supports the project hypothesis:

Current SGLang RadixAttention prefix caching can reduce prefill computation for repeated ordered prefixes, but it does not fully exploit reusable semantically independent sub-contexts when their order changes.

The next implementation step should add instrumentation and then prototype sub-context-aware cache lookup. A first patch should log the internal cache lookup behavior before changing semantics.

## Proposed Next Patch

Start with logging only:

- Request id
- Prompt token length
- Matched prefix length
- Cached token count
- New prefill token count
- Radix cache hit/miss
- Optional sub-context boundary metadata, once request format is defined

Likely SGLang source areas:

```text
python/sglang/srt/mem_cache/radix_cache.py
python/sglang/srt/managers/schedule_policy.py
python/sglang/srt/managers/scheduler.py
python/sglang/srt/layers/attention/
```

The later sub-context-aware KV cache design must handle causal attention and positional encoding carefully. KV entries for an arbitrary sub-context cannot be treated as independently reusable unless the implementation defines compatible attention masking and position handling.
