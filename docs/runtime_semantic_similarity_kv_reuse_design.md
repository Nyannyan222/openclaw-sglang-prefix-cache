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
3. substitute the canonical block into prompts,
4. let native prefix cache or exact semantic-block cache reuse the canonical tokens.

## New Protocol

Use:

```bash
python scripts/prepare_semantic_similarity_kv_reuse.py \
  --semantic-jsonl benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_semantic_subcontext_pilot.jsonl \
  --output-dir benchmark_results/semantic_similarity_kv_reuse \
  --threshold 0.18 \
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
- `similar_context`: low cache if token-different, even if similarity is high;
- `canonical_context_repeat`: high cache because the exact canonical text repeats.

The gap between `similar_context` and `canonical_context_repeat` motivates a
semantic similarity layer that canonicalizes or substitutes similar context
before runtime, rather than unsafe approximate KV tensor reuse.
