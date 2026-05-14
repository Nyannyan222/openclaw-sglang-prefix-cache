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

- `docs/subcontext_cache_prototype.md`
  Prototype design for layering a metadata-driven `SubContextIndex` outside
  SGLang's RadixAttention prefix cache.

- `docs/single_vs_subcontext_experiment.md`
  Experiment design for comparing No cache, native RadixAttention, and the
  sub-context-aware prototype over R1-R5.

- `docs/wildclaw_semantic_subcontext_pilot.md`
  First-week plan for replacing synthetic A/B/C contexts with real
  WildClawBench task workspaces and LLM semantic sub-context extraction.

- `docs/wildclaw_retrieval_execution_report.md`
  Report-ready summary of the Search Retrieval pilot after adding two-stage
  retrieval/search execution and combining the tuned task 4 rerun.

- `docs/wildclaw_phase2_roadmap.md`
  Phase-2 roadmap for expanding categories, adding formal grading, replaying
  through SGLang runtime, and preparing semantic sub-context KV reuse.

- `scripts/real_context_extractor.py`
  Builds leakage-checked long-context JSONL rows from WildClawBench task
  prompts and workspace files.

- `scripts/llm_semantic_segmenter.py`
  Extracts LLM semantic sub-contexts and writes fixed-size/file-based
  baselines plus a comparison CSV.

- `scripts/wildclaw_candidate_evidence_extractor.py`
  Builds V3 candidate-evidence contexts before LLM semantic segmentation.

- `scripts/prepare_wildclaw_next_experiment.py`
  Converts reviewed V3 outputs into a final pass-only dataset and a framework
  evaluation manifest comparing fixed-size, file-based, candidate-evidence, and
  V3 semantic sub-context policies.

- `scripts/run_wildclaw_framework_eval.py`
  Stages or runs the next-round WildClawBench framework-evaluation manifest,
  writing per-condition prompts, answers, retrieval evidence, token estimates,
  and result logs. The `openai-web` backend runs a two-stage retrieval/search
  execution path: focused web retrieval first, final answer second.

- `scripts/annotate_wildclaw_framework_eval.py`
  Adds first-pass correctness/evidence labels to a framework-evaluation run and
  writes a condition summary plus `run_summary_web.md`.

- `scripts/combine_wildclaw_eval_rerun.py`
  Replaces selected task rows from a base annotated run with a task-specific
  rerun, then writes a combined 12-row report.

- `scripts/prepare_wildclaw_phase2.py`
  Creates the phase-2 task-selection manifest, grading rubric, manual grading
  sheet, and SGLang runtime replay manifest.

- `scripts/prepare_wildclaw_mixed_category_eval.py`
  Builds a mixed-category WildClawBench manifest across Productivity, Search
  Retrieval, and Safety conditions. It supports prompt-only rows when a local
  workspace is absent, while preserving the same framework-eval runner shape.

- `scripts/run_wildclaw_sglang_runtime_replay.py`
  Replays WildClaw prompts through an SGLang OpenAI-compatible endpoint and
  records cached-token, prefill-token estimate, and latency metrics. It can
  flush SGLang cache before each row/replay/pair for isolated cache experiments.

- `scripts/prepare_semantic_nonprefix_kv_reuse.py`
  Generates runtime prompts and a semantic block manifest for measuring
  non-prefix KV reuse opportunity with reordered semantic sub-contexts.

- `scripts/find_semantic_similar_subcontexts.py`
  Finds genuinely semantically similar WildClaw sub-context pairs/groups.
  Lexical overlap is only a prefilter; OpenAI embeddings and optional LLM judge
  provide the semantic signal when `OPENAI_API_KEY` is available. Strict mode
  requires judge agreement, enough embedding similarity, and same-answer utility.

- `scripts/start_local_sglang_docker.ps1`
  Starts the local Docker Desktop SGLang runtime on the RTX 5070 using
  `lmsysorg/sglang:latest-runtime` and `Qwen/Qwen2.5-0.5B-Instruct`.

- `docs/runtime_nonprefix_kv_reuse_design.md`
  Runtime hook contract for moving from native prefix cache observation toward
  semantic non-prefix KV reuse inside SGLang.

- `docs/semantic_subcontext_similarity_discovery.md`
  Current direction for finding genuinely semantically similar WildClawBench
  sub-context pairs/groups using embeddings, LLM judge, and manual review.

- `docs/local_rtx5070_neno5_execution_plan.md`
  Execution split for running data preparation and reports on the local RTX
  5070 machine while sending SGLang runtime replay jobs to neno5 when the local
  runtime is unavailable.

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

## Sub-Context Index Prototype

After running a benchmark, simulate segment-level reuse over the exported
sub-context metadata:

```bash
python3 scripts/subcontext_cache_prototype.py \
  benchmark_results/<run_dir>/sglang_prefix_cache_<timestamp>.json
```

This writes `subcontext_cache_prototype_<timestamp>.csv/json`. It does not
modify SGLang or splice real KV cache blocks; it records which A/B/C spans
would be hit by a `hash(sub-context) -> token_range, kv_block_refs` index.

## Single-Context vs Sub-Context Matrix

Run the WSL comparison matrix:

```bash
bash scripts/wsl_run_cache_baseline_matrix.sh
```

It runs:

```text
Baseline_NoCache
Baseline_RadixAttention
Proposed_SubContextIndex
```

and exports the four comparison metrics:

```text
cached_token_ratio
prefill_tokens
first_token_latency_s
total_latency_s
```



Do not start SGLang directly on the login node. The SLURM script runs it inside
a GPU job and installs a local Node.js runtime under `/work/$USER/openclaw-sglang/node`
if the cluster does not provide `node`/`npm`.
