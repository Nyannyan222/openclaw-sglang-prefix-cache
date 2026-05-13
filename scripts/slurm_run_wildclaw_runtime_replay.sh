#!/usr/bin/env bash
# Run WildClaw phase-2 replay against a real SGLang server inside a SLURM GPU job.
#
# Submit from the repository root:
#   sbatch --account=<project_id> scripts/slurm_run_wildclaw_runtime_replay.sh
#
# Useful overrides:
#   sbatch --account=<project_id> --export=ALL,LIMIT=12,REPEAT=2 scripts/slurm_run_wildclaw_runtime_replay.sh
#   sbatch --account=<project_id> --export=ALL,MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct,LIMIT=12 scripts/slurm_run_wildclaw_runtime_replay.sh

#SBATCH --job-name=wildclaw-sglang-replay
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%j-wildclaw-sglang-replay.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORK_ROOT="${WORK_ROOT:-/work/$USER/openclaw-sglang}"
RUNTIME_DIR="$WORK_ROOT/runtime"
LOG_DIR="$WORK_ROOT/logs"
HF_HOME_DIR="$WORK_ROOT/huggingface"
NPM_PREFIX="$WORK_ROOT/npm"
NODE_INSTALL_DIR="$WORK_ROOT/node"
UV_BIN_DIR="$WORK_ROOT/uv/bin"
UV_CACHE_DIR="$WORK_ROOT/uv-cache"
UV_PYTHON_INSTALL_DIR="$WORK_ROOT/uv-python"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_LABEL="${MODEL_LABEL:-${MODEL_ID//\//__}}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_LOG="$LOG_DIR/wildclaw_sglang_${MODEL_LABEL}_${SLURM_JOB_ID:-manual}.log"
MANIFEST="${MANIFEST:-benchmark_results/wildclaw_phase2/wildclaw_phase2_sglang_runtime_manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results/wildclaw_sglang_runtime_runs}"
REPEAT="${REPEAT:-2}"
LIMIT="${LIMIT:-2}"
MAX_TOKENS="${MAX_TOKENS:-64}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
METRICS_TIMEOUT="${METRICS_TIMEOUT:-5}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.45}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

export PATH="$UV_BIN_DIR:$NODE_INSTALL_DIR/bin:$NPM_PREFIX/bin:$HOME/.local/bin:$PATH"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-false}"
export SGLANG_PREFIX_CACHE_DEBUG_LOG="${SGLANG_PREFIX_CACHE_DEBUG_LOG:-1}"

if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

try_module_load() {
  for module_name in "$@"; do
    if command -v module >/dev/null 2>&1 && module load "$module_name" >/dev/null 2>&1; then
      echo "Loaded module: $module_name"
      return 0
    fi
  done
  return 1
}

if [ -n "${CUDA_MODULE:-}" ]; then
  try_module_load "$CUDA_MODULE" || {
    echo "ERROR: requested CUDA_MODULE=$CUDA_MODULE could not be loaded."
    exit 1
  }
fi

if [ -n "${CXX_MODULE:-}" ]; then
  try_module_load "$CXX_MODULE" || {
    echo "ERROR: requested CXX_MODULE=$CXX_MODULE could not be loaded."
    exit 1
  }
fi

cuda_home_ok() {
  [ -n "${CUDA_HOME:-}" ] &&
    [ -d "$CUDA_HOME" ] &&
    [ -d "$CUDA_HOME/lib64" ] &&
    { [ -x "$CUDA_HOME/bin/nvcc" ] || command -v nvcc >/dev/null 2>&1; }
}

set_cuda_home_from_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    export CUDA_HOME
  fi
}

set_cuda_home_from_candidates() {
  local candidate
  for candidate in \
    /usr/local/cuda \
    /usr/local/cuda-12.8 \
    /usr/local/cuda-12.6 \
    /usr/local/cuda-12.4 \
    /opt/cuda \
    /opt/cuda-12.8 \
    /opt/cuda-12.6 \
    /opt/cuda-12.4; do
    if [ -d "$candidate" ]; then
      CUDA_HOME="$candidate"
      export CUDA_HOME
      return 0
    fi
  done
  return 1
}

ensure_cuda_home() {
  if cuda_home_ok; then
    return 0
  fi

  try_module_load cuda/12.8 cuda/12.6 cuda/12.4 cuda/12 cuda cudatoolkit/12.8 cudatoolkit/12.6 cudatoolkit/12.4 cudatoolkit || true
  set_cuda_home_from_nvcc
  if ! cuda_home_ok; then
    set_cuda_home_from_candidates || true
  fi

  if ! cuda_home_ok; then
    echo "ERROR: CUDA toolkit was not found. SGLang JIT needs CUDA_HOME and nvcc."
    echo "Try checking CUDA modules:"
    echo "  module avail cuda"
    echo "Then run, for example:"
    echo "  sbatch --account=<project_id> --export=ALL,CUDA_MODULE=<module-name> scripts/slurm_run_wildclaw_runtime_replay.sh"
    exit 1
  fi

  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
}

cpp20_toolchain_ok() {
  command -v nvcc >/dev/null 2>&1 || return 1
  command -v "${CXX:-g++}" >/dev/null 2>&1 || return 1

  local tmp_dir
  tmp_dir="$(mktemp -d "$WORK_ROOT/cxx20-check.XXXXXX")"
  cat > "$tmp_dir/check.cu" <<'EOF'
#include <version>
int main() { return 0; }
EOF
  if nvcc -std=c++20 -ccbin "${CXX:-g++}" -c "$tmp_dir/check.cu" -o "$tmp_dir/check.o" >/dev/null 2>&1; then
    rm -rf "$tmp_dir"
    return 0
  fi
  echo "C++20 CUDA compile check failed. Compiler details:"
  "${CXX:-g++}" --version | sed -n '1,3p' || true
  nvcc -std=c++20 -ccbin "${CXX:-g++}" -c "$tmp_dir/check.cu" -o "$tmp_dir/check.o" || true
  rm -rf "$tmp_dir"
  return 1
}

ensure_cpp20_toolchain() {
  if cpp20_toolchain_ok; then
    return 0
  fi

  try_module_load gcc/13 gcc/12 gcc/11 gcc/10 gcc/9 gcc cuda-gcc/12 cuda-gcc/11 || true

  if command -v g++ >/dev/null 2>&1; then
    CXX="$(command -v g++)"
    CC="$(command -v gcc || echo gcc)"
    CUDAHOSTCXX="$CXX"
    export CC CXX CUDAHOSTCXX
  fi

  if ! cpp20_toolchain_ok; then
    echo "ERROR: CUDA JIT needs a C++20-capable host compiler with the <version> header."
    echo "Try checking compiler modules:"
    echo "  module avail gcc"
    echo "Then run, for example:"
    echo "  sbatch --account=<project_id> --export=ALL,CUDA_MODULE=cuda/12.4,CXX_MODULE=<module-name> scripts/slurm_run_wildclaw_runtime_replay.sh"
    exit 1
  fi
}

if [ ! -x "$RUNTIME_DIR/.venv/bin/python" ]; then
  echo "ERROR: SGLang runtime venv not found at $RUNTIME_DIR/.venv"
  echo "Run setup first:"
  echo "  sbatch --account=<project_id> scripts/slurm_setup_env.sh"
  exit 1
fi

if [ ! -f "$PROJECT_ROOT/$MANIFEST" ]; then
  echo "ERROR: manifest not found: $PROJECT_ROOT/$MANIFEST"
  exit 1
fi

cleanup() {
  if [ -n "${SGLANG_PID:-}" ] && kill -0 "$SGLANG_PID" >/dev/null 2>&1; then
    echo "Stopping SGLang PID $SGLANG_PID"
    kill "$SGLANG_PID" || true
    wait "$SGLANG_PID" || true
  fi
}
trap cleanup EXIT

echo "== Job context =="
date
hostname
whoami
nvidia-smi || true
echo "Project root: $PROJECT_ROOT"
echo "Runtime dir:  $RUNTIME_DIR"
echo "Model:        $MODEL_ID"
echo "Manifest:     $MANIFEST"
echo "Limit:        $LIMIT"
echo "Repeat:       $REPEAT"
echo

echo "== Ensure CUDA toolkit for SGLang JIT =="
ensure_cuda_home
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version | sed -n '1,4p'
echo

echo "== Ensure C++20 compiler for SGLang JIT =="
ensure_cpp20_toolchain
echo "CC=${CC:-}"
echo "CXX=${CXX:-}"
"${CXX:-g++}" --version | sed -n '1,3p'
echo

echo "== Start SGLang =="
cd "$RUNTIME_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
: > "$SGLANG_LOG"

python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$SGLANG_PORT" \
  --enable-metrics \
  --log-requests \
  --log-requests-level 1 \
  --log-requests-format json \
  --radix-eviction-policy lru \
  --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  > "$SGLANG_LOG" 2>&1 &

SGLANG_PID=$!
echo "SGLang PID=$SGLANG_PID"
echo "SGLang log=$SGLANG_LOG"

echo "== Wait for SGLang readiness =="
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; then
    echo "SGLang is ready after ${i} checks."
    break
  fi
  if ! kill -0 "$SGLANG_PID" >/dev/null 2>&1; then
    echo "ERROR: SGLang exited early. Last log lines:"
    tail -n 180 "$SGLANG_LOG" || true
    exit 1
  fi
  sleep 10
  if [ "$i" = "120" ]; then
    echo "ERROR: timed out waiting for SGLang. Last log lines:"
    tail -n 180 "$SGLANG_LOG" || true
    exit 1
  fi
done

curl -sS "http://127.0.0.1:${SGLANG_PORT}/v1/models"
echo
curl -sS "http://127.0.0.1:${SGLANG_PORT}/metrics" | grep -E "cache|token|request" | head -n 80 || true
echo

echo "== Run WildClaw runtime replay =="
cd "$PROJECT_ROOT"

replay_args=(
  scripts/run_wildclaw_sglang_runtime_replay.py
  --manifest "$MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --base-url "http://127.0.0.1:${SGLANG_PORT}/v1"
  --metrics-url "http://127.0.0.1:${SGLANG_PORT}/metrics"
  --model "$MODEL_ID"
  --repeat "$REPEAT"
  --max-tokens "$MAX_TOKENS"
  --timeout "$REQUEST_TIMEOUT"
  --metrics-timeout "$METRICS_TIMEOUT"
)

if [ "$LIMIT" != "0" ]; then
  replay_args+=(--limit "$LIMIT")
fi

"$RUNTIME_DIR/.venv/bin/python" "${replay_args[@]}"

echo
echo "== Latest WildClaw runtime result =="
latest_run="$(ls -td "$OUTPUT_DIR"/wildclaw_sglang_runtime_* | head -1)"
echo "$latest_run"
find "$latest_run" -maxdepth 1 -type f -print
echo
echo "== Cache log sample =="
grep -E "#cached-token|cached_tokens|request.finished|cache_lookup" "$SGLANG_LOG" | tail -n 80 || true
