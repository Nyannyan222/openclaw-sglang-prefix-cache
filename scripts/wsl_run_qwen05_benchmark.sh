#!/usr/bin/env bash
# Start a local WSL SGLang server for Qwen2.5-0.5B and run R1-R5 benchmark.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/c/Users/Administrator/OneDrive/文件/New project}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/chihjou/openclaw-sglang-wsl}"
VENV_DIR="${VENV_DIR:-$RUNTIME_ROOT/.venv}"
CUDA_TOOLKIT_DIR="${CUDA_TOOLKIT_DIR:-/home/chihjou/cuda-12.8-nvcc}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${SGLANG_PORT:-31080}"
CACHE_MODE="${CACHE_MODE:-radix}"
LOG_DIR="$RUNTIME_ROOT/logs"
SGLANG_LOG="$LOG_DIR/sglang_qwen05_wsl_${CACHE_MODE}.log"
RESULT_DIR="$PROJECT_ROOT/benchmark_results/wsl_${CACHE_MODE}_Qwen__Qwen2.5-0.5B-Instruct_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

cd "$PROJECT_ROOT"
source "$VENV_DIR/bin/activate"

python scripts/patch_sglang_cache_lookup_logging.py

export SGLANG_PREFIX_CACHE_DEBUG_LOG="${SGLANG_PREFIX_CACHE_DEBUG_LOG:-1}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-false}"
export HF_HOME="${HF_HOME:-/home/chihjou/.cache/huggingface}"
if [[ -x "$CUDA_TOOLKIT_DIR/bin/nvcc" ]]; then
  export CUDA_HOME="$CUDA_TOOLKIT_DIR"
  export CUDACXX="$CUDA_TOOLKIT_DIR/bin/nvcc"
  export PATH="$CUDA_TOOLKIT_DIR/bin:$PATH"
fi

cleanup() {
  if [[ -n "${SGLANG_PID:-}" ]] && kill -0 "$SGLANG_PID" >/dev/null 2>&1; then
    echo "Stopping SGLang PID $SGLANG_PID"
    kill "$SGLANG_PID" || true
    wait "$SGLANG_PID" || true
  fi
}
trap cleanup EXIT

: > "$SGLANG_LOG"
echo "Starting SGLang"
echo "Model: $MODEL_ID"
echo "Cache: $CACHE_MODE"
echo "Port : $PORT"
echo "Log  : $SGLANG_LOG"

SERVER_CACHE_ARGS=(--radix-eviction-policy lru)
case "$CACHE_MODE" in
  no_cache)
    SERVER_CACHE_ARGS+=(--disable-radix-cache)
    ;;
  radix)
    ;;
  *)
    echo "Unsupported CACHE_MODE=$CACHE_MODE. Use no_cache or radix."
    exit 2
    ;;
esac

python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --enable-metrics \
  --log-requests \
  --log-requests-level 1 \
  --log-requests-format json \
  "${SERVER_CACHE_ARGS[@]}" \
  --mem-fraction-static 0.35 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  > "$SGLANG_LOG" 2>&1 &

SGLANG_PID=$!
echo "SGLang PID=$SGLANG_PID"

echo "Waiting for SGLang readiness"
for i in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "SGLang ready after $i checks"
    break
  fi
  if ! kill -0 "$SGLANG_PID" >/dev/null 2>&1; then
    echo "SGLang exited early"
    tail -n 180 "$SGLANG_LOG" || true
    exit 1
  fi
  sleep 2
  if [[ "$i" == "90" ]]; then
    echo "Timed out waiting for SGLang"
    tail -n 180 "$SGLANG_LOG" || true
    exit 1
  fi
done

echo "Running benchmark"
python bench_sglang_prefix_cache.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" \
  --log-file "$SGLANG_LOG" \
  --output-dir "$RESULT_DIR" \
  --timeout 180 \
  --max-tokens 64 \
  --sleep-after-request 0.5

echo
echo "Results:"
find "$RESULT_DIR" -maxdepth 1 -type f -print
echo
echo "Cache lookup log sample:"
grep -E '"event":"cache_lookup"|request.finished|cached_tokens' "$SGLANG_LOG" | tail -n 80 || true
