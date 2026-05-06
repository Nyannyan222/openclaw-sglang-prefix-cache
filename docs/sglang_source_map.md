# SGLang Cache/Logging Source Map

This note identifies the first source locations to inspect before changing cache semantics.

Source checkout:

```text
external/sglang
```

Checked commit:

```text
660a77f22
```

The checkout is intentionally ignored by the parent repository because it is a full external source tree.

## Prefix Cache Lookup

Primary entry:

```text
external/sglang/python/sglang/srt/managers/schedule_policy.py
```

Important functions:

- `match_prefix_for_req(...)`, around line 80
- `SchedulePolicy._compute_prefix_matches(...)`, around line 221
- `PrefillAdder._update_prefill_budget(...)`, around line 565

Current flow:

```text
Req origin_input_ids + output_ids
  -> RadixKey(token_ids=..., extra_key=req.extra_key)
  -> tree_cache.match_prefix(...)
  -> req.prefix_indices / req.last_node / req.host_hit_length
```

`PrefillAdder._update_prefill_budget()` increments:

```text
log_hit_tokens += prefix_len
log_input_tokens += extend_input_len
```

These values become the high-level `#cached-token` and `#new-token` logs.

## Radix Tree Implementation

Primary file:

```text
external/sglang/python/sglang/srt/mem_cache/radix_cache.py
```

Important classes/functions:

- `RadixKey`, around line 71
- `RadixKey.match(...)`, around line 152
- `RadixCache.match_prefix(...)`, around line 398
- `RadixCache.insert(...)`, around line 468
- `RadixCache._match_prefix_helper(...)`, around line 693

Important current behavior:

`RadixKey` matches by token prefix plus `extra_key`. If token ids diverge early, matching stops. This is exactly why reordered sub-contexts have poor reuse in the baseline.

The `extra_key` field is useful for cache namespacing, but it does not solve reordered sub-context reuse by itself. It can isolate caches, not make non-prefix spans reusable.

## Existing Metrics/Logs

Primary file:

```text
external/sglang/python/sglang/srt/observability/scheduler_metrics_mixin.py
```

Important area:

- `_log_prefill_stats(...)`, around lines 378-443

This is where SGLang prints:

```text
Prefill batch, #new-token: ..., #cached-token: ...
```

It also updates:

```text
sglang:realtime_tokens_total{mode="prefill_compute"}
sglang:realtime_tokens_total{mode="prefill_cache"}
sglang:cache_hit_rate
```

## Per-Request Cached Token Output

Primary files:

```text
external/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py
external/sglang/python/sglang/srt/managers/tokenizer_manager.py
external/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
external/sglang/python/sglang/srt/entrypoints/openai/usage_processor.py
```

Useful locations:

- `scheduler_output_processor_mixin.py`: `_get_cached_tokens_details(...)`, and construction of scheduler outputs with `cached_tokens=...`
- `tokenizer_manager.py`: places scheduler `cached_tokens` into `meta_info`
- `serving_chat.py`: converts `meta_info` into OpenAI-compatible usage fields
- `usage_processor.py`: creates OpenAI `prompt_tokens_details.cached_tokens`

## Recommended First Patch

Do not change cache semantics first. Add logging around prefix lookup:

```text
schedule_policy.match_prefix_for_req()
```

Proposed debug event fields:

```text
rid
token_count
extra_key
matched_prefix_len = len(req.prefix_indices)
host_hit_length
last_node_present
cache_protected_len
```

This should be guarded by a flag or environment variable, for example:

```text
SGLANG_PREFIX_CACHE_DEBUG=1
```

## Recommended Second Patch

Add optional request metadata for sub-context boundaries, but keep it logging-only:

```json
{
  "subcontexts": [
    {"id": "A", "start": 10, "end": 120},
    {"id": "B", "start": 121, "end": 240},
    {"id": "C", "start": 241, "end": 360}
  ]
}
```

The first prototype should only log whether each sub-context token span has appeared before under a stable hash:

```text
subcontext_id
token_start
token_end
token_len
token_hash
seen_before
```

This avoids making unsafe KV reuse assumptions before attention masking and positional encoding are handled.
