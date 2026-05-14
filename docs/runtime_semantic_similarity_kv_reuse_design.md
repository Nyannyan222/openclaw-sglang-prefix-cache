# Runtime-Level Semantic / Content Similarity Reuse Design

## Goal

This direction does not focus on reordered exact blocks. It focuses on detecting
sub-contexts that are semantically or content-wise similar, then deciding whether
they can reduce runtime cost.

## Key Constraint

KV cache is not a semantic memory. KV tensors depend on exact token ids,
positions, model weights, tokenizer, system prefix, and attention history.
Therefore:

- exact same text/token ids: direct KV reuse can be safe under normal cache constraints;
- normalized-identical text: canonicalize first, then reuse the canonical text;
- semantically similar but token-different text: do not directly splice KV tensors.

For semantically similar text, the safe path is:

1. detect similarity,
2. choose or create a canonical context block,
3. extract task-specific delta from the similar context,
4. build `canonical context + delta` prompts,
5. let native prefix cache or exact semantic-block cache reuse the canonical tokens.

This method is `Semantic Sub-context Canonicalization with Delta Preservation`.

## New Protocol

Use:

```bash
python scripts/prepare_semantic_similarity_kv_reuse.py \
  --semantic-jsonl benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_semantic_subcontext_pilot.jsonl \
  --output-dir benchmark_results/semantic_similarity_kv_reuse \
  --threshold 0.05 \
  --max-pairs 8
```

It writes:

- `semantic_similarity_pairs.csv`: similar sub-context pairs and reuse policy.
- `semantic_similarity_runtime_manifest.jsonl`: SGLang replay manifest.
- `semantic_similarity_block_manifest.jsonl`: block hashes and normalized hashes.
- `semantic_similarity_protocol.json`: runtime safety contract.

## Measurement

Run the generated manifest through SGLang:

```bash
sbatch --account=MST114180 \
  --export=ALL,MANIFEST=benchmark_results/semantic_similarity_kv_reuse/semantic_similarity_runtime_manifest.jsonl,LIMIT=0,REPEAT=1 \
  scripts/slurm_run_wildclaw_runtime_replay.sh
```

Expected native SGLang behavior:

- `canonical_context`: low cache on first use;
- `similar_context`: only tiny fixed-marker cache if token-different, even if similarity is high;
- `canonical_plus_delta`: higher cache than `similar_context` because it starts with exact canonical tokens, while still preserving task-specific delta;
- `canonical_context_repeat`: high cache because the exact canonical text appears first and repeats.

The key comparison is `similar_context` vs `canonical_plus_delta`. If
`canonical_plus_delta` has higher cached-token ratio and lower prefill while
preserving answer quality, then the proposed method improves runtime reuse
without unsafe token-different KV tensor splicing.

The generated prompts are context-first by design. The first tokens are the
candidate context block rather than a long shared instruction prefix, which makes
`cached_tokens` easier to interpret as context reuse instead of prompt-template
reuse.

For `canonical_plus_delta`, the prompt intentionally starts with the exact same
token prefix as `canonical_context` and appends `DELTA_BLOCK` afterward. This is
necessary because SGLang's native prefix cache only matches exact prefixes.
