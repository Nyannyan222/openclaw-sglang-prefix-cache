# Runtime-Level Non-Prefix KV Reuse Design

## Goal

Native SGLang RadixAttention reuses KV cache for repeated prefixes. Our target is different: reuse semantically identical sub-context blocks even when their order changes, such as `A+B+C` followed by `C+A+B`.

## Current Implementation State

This repository now prepares the runtime protocol but does not yet splice KV tensors inside SGLang. The prepared artifacts are:

- `semantic_nonprefix_runtime_manifest.jsonl`: prompts for order-A, non-prefix order-B, and order-A repeat.
- `semantic_kv_block_manifest.jsonl`: block-level content hashes and expected runtime action.
- `semantic_nonprefix_protocol.json`: the runtime hook contract.

## Runtime Hook Contract

The future SGLang patch needs to run after tokenization and before prefill allocation:

1. Parse or receive semantic block boundaries.
2. Compute a stable block hash from canonical block text and token ids.
3. Look up `hash -> KV block refs`.
4. Validate same model, tokenizer, dtype, system prefix, tenant boundary, and adapter state.
5. Reuse only when tokenization and attention dependencies are safe.
6. For reordered blocks, either remap RoPE positions correctly or initially restrict reuse to controlled validation runs.

## Measurement

Use the generated runtime manifest with `scripts/run_wildclaw_sglang_runtime_replay.py`:

```bash
python scripts/run_wildclaw_sglang_runtime_replay.py \
  --manifest benchmark_results/semantic_nonprefix_kv_reuse/semantic_nonprefix_runtime_manifest.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --metrics-url http://127.0.0.1:30000/metrics \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --repeat 1 \
  --max-tokens 64
```

Expected native-prefix result:

- order-A repeat should show high prefix-cache reuse.
- order-B non-prefix should show much lower native prefix reuse.
- The gap is the runtime-level semantic KV reuse opportunity.
