# Sub-Context-Aware Cache Prototype

This prototype keeps SGLang's RadixAttention prefix cache as the baseline and
adds a metadata-driven index outside the radix tree:

```text
SubContextIndex:
  hash(A) -> token_range, kv_block_refs
  hash(B) -> token_range, kv_block_refs
  hash(C) -> token_range, kv_block_refs
```

For a new request such as `R3 = C + A + B`, the prototype flow is:

1. Tokenize the prompt.
2. Read sub-context metadata for `C`, `A`, and `B`.
3. Compute or read each sub-context content hash.
4. Query `SubContextIndex` by hash.
5. Mark hit spans as reusable candidates.
6. Send miss spans through normal prefill.

This changes the experiment target from full prefix reuse to segment-level
reuse opportunities. It is intentionally not a full radix-tree rewrite.

## What The Prototype Does

`scripts/subcontext_cache_prototype.py` consumes the JSON output from
`bench_sglang_prefix_cache.py`, uses the existing `subcontext_metadata`, and
simulates a `SubContextIndex` over request order.

Example:

```bash
python scripts/subcontext_cache_prototype.py \
  benchmark_results/wsl_Qwen__Qwen2.5-0.5B-Instruct_20260507_191531/sglang_prefix_cache_20260507_191550.json
```

It writes:

```text
subcontext_cache_prototype_<timestamp>.csv
subcontext_cache_prototype_<timestamp>.json
```

The output records each sub-context span with:

```text
request_name
subcontext_id
content_hash
token_start / token_end / token_len
status = hit | miss
reusable_tokens
prefill_tokens
source_request_name
kv_block_refs
```

The summary compares prototype segment reuse against SGLang's observed
`lookup_matched_prefix_len`.

## Important Limitation

The prototype only proves that metadata can identify reusable sub-context
spans. It does not prove that arbitrary KV blocks can be spliced safely.

For a production runtime, KV reuse across reordered spans needs explicit
handling for:

- absolute or rotary position differences,
- causal attention dependencies on earlier tokens,
- block ownership and lifetime,
- correctness checks when prompt separators or templates change.

So the safe claim is:

```text
We can layer a SubContextIndex outside RadixAttention to detect and mark
segment-level reuse candidates. Actual KV reuse requires runtime support for
position and attention-dependency handling.
```
