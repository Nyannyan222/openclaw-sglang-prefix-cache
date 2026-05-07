#!/usr/bin/env bash
# Run No cache, RadixAttention, and proposed SubContextIndex comparison in WSL.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/c/Users/Administrator/OneDrive/文件/New project}"
PORT="${SGLANG_PORT:-31080}"
OUTPUT_ROOT="$PROJECT_ROOT/benchmark_results/wsl_cache_baseline_matrix_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

echo "== Baseline 1: No cache =="
CACHE_MODE=no_cache SGLANG_PORT="$PORT" bash scripts/wsl_run_qwen05_benchmark.sh
NO_CACHE_JSON="$(ls -t benchmark_results/wsl_no_cache_Qwen__Qwen2.5-0.5B-Instruct_*/sglang_prefix_cache_*.json | head -n 1)"
echo "NO_CACHE_JSON=$NO_CACHE_JSON" | tee "$OUTPUT_ROOT/inputs.txt"

echo
echo "== Baseline 2: SGLang RadixAttention =="
CACHE_MODE=radix SGLANG_PORT="$PORT" bash scripts/wsl_run_qwen05_benchmark.sh
RADIX_JSON="$(ls -t benchmark_results/wsl_radix_Qwen__Qwen2.5-0.5B-Instruct_*/sglang_prefix_cache_*.json | head -n 1)"
echo "RADIX_JSON=$RADIX_JSON" | tee -a "$OUTPUT_ROOT/inputs.txt"

echo
echo "== Proposed: Sub-context-aware cache prototype =="
python3 scripts/subcontext_cache_prototype.py "$RADIX_JSON" --output-dir "$OUTPUT_ROOT"
PROPOSED_JSON="$(ls -t "$OUTPUT_ROOT"/subcontext_cache_prototype_*.json | head -n 1)"
echo "PROPOSED_JSON=$PROPOSED_JSON" | tee -a "$OUTPUT_ROOT/inputs.txt"

echo
echo "== Combined comparison =="
python3 scripts/compare_cache_baselines.py \
  --no-cache-json "$NO_CACHE_JSON" \
  --radix-json "$RADIX_JSON" \
  --proposed-json "$PROPOSED_JSON" \
  --output-dir "$OUTPUT_ROOT"

echo
echo "Matrix output:"
find "$OUTPUT_ROOT" -maxdepth 1 -type f -print
