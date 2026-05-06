#!/usr/bin/env bash
# Run OpenClaw + SGLang setup and the prefix-cache benchmark inside a SLURM GPU job.
#
# Submit from the repository root:
#   sbatch --account=<project_id> scripts/slurm_setup_and_benchmark.sh
#
# If your site requires a different partition/account, override at submission:
#   sbatch -p <partition> -A <account> --time=00:30:00 scripts/slurm_setup_and_benchmark.sh

#SBATCH --job-name=oc-sglang-cache
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-openclaw-sglang.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORK_ROOT="${WORK_ROOT:-/work/$USER/openclaw-sglang}"
RUNTIME_DIR="$WORK_ROOT/runtime"
LOG_DIR="$WORK_ROOT/logs"
HF_HOME_DIR="$WORK_ROOT/huggingface"
NPM_PREFIX="$WORK_ROOT/npm"
NODE_INSTALL_DIR="$WORK_ROOT/node"
SGLANG_LOG="$LOG_DIR/sglang_openclaw_${SLURM_JOB_ID:-manual}.log"
RESULT_DIR="$PROJECT_ROOT/benchmark_results/neno5_${SLURM_JOB_ID:-manual}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NODE_VERSION="${NODE_VERSION:-v20.19.0}"

mkdir -p "$RUNTIME_DIR" "$LOG_DIR" "$HF_HOME_DIR" "$NPM_PREFIX" "$NODE_INSTALL_DIR" "$RESULT_DIR"

export PATH="$NODE_INSTALL_DIR/bin:$NPM_PREFIX/bin:$HOME/.local/bin:$PATH"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ -f /etc/profile.d/modules.sh ]; then
  # Many HPC systems expose Node/Python/CUDA through environment modules.
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

if [ -n "${NODE_MODULE:-}" ]; then
  try_module_load "$NODE_MODULE" || {
    echo "ERROR: requested NODE_MODULE=$NODE_MODULE could not be loaded."
    exit 1
  }
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  try_module_load nodejs node node/20 nodejs/20 node/22 nodejs/22 npm || true
fi

python_version_ok() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if sys.version_info >= (3, 9) else 1)
PY
}

select_python_bin() {
  if [ -n "${PYTHON_MODULE:-}" ]; then
    try_module_load "$PYTHON_MODULE" || {
      echo "ERROR: requested PYTHON_MODULE=$PYTHON_MODULE could not be loaded."
      exit 1
    }
  fi

  local candidate
  for candidate in "$PYTHON_BIN" python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && python_version_ok "$candidate"; then
      PYTHON_BIN="$(command -v "$candidate")"
      export PYTHON_BIN
      echo "Selected Python: $PYTHON_BIN"
      return 0
    fi
  done

  try_module_load \
    python/3.12 python/3.11 python/3.10 python/3.9 \
    python3/3.12 python3/3.11 python3/3.10 python3/3.9 \
    python312 python311 python310 python39 python || true

  for candidate in "$PYTHON_BIN" python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && python_version_ok "$candidate"; then
      PYTHON_BIN="$(command -v "$candidate")"
      export PYTHON_BIN
      echo "Selected Python: $PYTHON_BIN"
      return 0
    fi
  done

  echo "ERROR: Python 3.9+ is required, but no suitable Python was found."
  echo "Try checking available modules on the login node with:"
  echo "  module avail python"
  echo "Then submit with, for example:"
  echo "  sbatch --export=ALL,PYTHON_MODULE=<module-name> scripts/slurm_setup_and_benchmark.sh"
  exit 1
}

select_python_bin

install_local_node() {
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    return 0
  fi

  local arch
  local node_arch
  arch="$(uname -m)"
  case "$arch" in
    x86_64) node_arch="linux-x64" ;;
    aarch64|arm64) node_arch="linux-arm64" ;;
    *)
      echo "ERROR: unsupported architecture for automatic Node.js install: $arch"
      return 1
      ;;
  esac

  local tarball="node-${NODE_VERSION}-${node_arch}.tar.xz"
  local url="https://nodejs.org/dist/${NODE_VERSION}/${tarball}"
  local download_dir="$WORK_ROOT/downloads"
  mkdir -p "$download_dir"

  if [ ! -x "$NODE_INSTALL_DIR/bin/node" ]; then
    echo "Node.js not found; installing local Node.js ${NODE_VERSION} into $NODE_INSTALL_DIR"
    if command -v curl >/dev/null 2>&1; then
      curl -fL "$url" -o "$download_dir/$tarball"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$download_dir/$tarball" "$url"
    else
      echo "ERROR: neither curl nor wget is available to download Node.js."
      return 1
    fi

    rm -rf "$NODE_INSTALL_DIR.tmp"
    mkdir -p "$NODE_INSTALL_DIR.tmp"
    tar -xJf "$download_dir/$tarball" -C "$NODE_INSTALL_DIR.tmp" --strip-components=1
    rm -rf "$NODE_INSTALL_DIR"
    mv "$NODE_INSTALL_DIR.tmp" "$NODE_INSTALL_DIR"
  fi

  export PATH="$NODE_INSTALL_DIR/bin:$PATH"
}

install_local_node

echo "== Job context =="
date
hostname
whoami
pwd
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "WORK_ROOT=$WORK_ROOT"
echo "MODEL_ID=$MODEL_ID"
echo

echo "== GPU =="
nvidia-smi || true
echo

echo "== Tool versions =="
"$PYTHON_BIN" --version
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: node/npm not found."
  echo "The script tried environment modules and local Node.js installation."
  echo "Try checking available modules on the login node with:"
  echo "  module avail node"
  echo "Then submit with, for example:"
  echo "  sbatch --export=ALL,NODE_MODULE=<module-name> scripts/slurm_setup_and_benchmark.sh"
  exit 1
fi
node --version
npm --version
echo

echo "== Install uv if needed =="
if ! command -v uv >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install --user uv
fi
uv --version
echo

echo "== Install OpenClaw if needed =="
if ! command -v openclaw >/dev/null 2>&1; then
  npm install -g openclaw@latest --prefix "$NPM_PREFIX"
fi
openclaw --version
echo

echo "== Create/update SGLang venv =="
cd "$RUNTIME_DIR"
if [ ! -d .venv ]; then
  uv venv --python "$PYTHON_BIN" .venv
fi
source .venv/bin/activate
uv pip install -U sglang hf_transfer
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
echo

cleanup() {
  if [ -n "${SGLANG_PID:-}" ] && kill -0 "$SGLANG_PID" >/dev/null 2>&1; then
    echo "Stopping SGLang PID $SGLANG_PID"
    kill "$SGLANG_PID" || true
    wait "$SGLANG_PID" || true
  fi
}
trap cleanup EXIT

echo "== Start SGLang =="
cd "$RUNTIME_DIR"
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
  --mem-fraction-static 0.65 \
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
    echo "SGLang exited early. Last log lines:"
    tail -n 160 "$SGLANG_LOG" || true
    exit 1
  fi
  sleep 10
  if [ "$i" = "120" ]; then
    echo "Timed out waiting for SGLang. Last log lines:"
    tail -n 160 "$SGLANG_LOG" || true
    exit 1
  fi
done

curl -sS "http://127.0.0.1:${SGLANG_PORT}/v1/models"
echo
curl -sS "http://127.0.0.1:${SGLANG_PORT}/metrics" | grep -E "cache|token|request" | head -n 80 || true
echo

echo "== Configure OpenClaw provider =="
openclaw onboard \
  --non-interactive \
  --accept-risk \
  --mode local \
  --auth-choice custom-api-key \
  --custom-provider-id sglang \
  --custom-base-url "http://127.0.0.1:${SGLANG_PORT}/v1" \
  --custom-model-id "$MODEL_ID" \
  --custom-api-key sglang-local \
  --custom-text-input \
  --gateway-auth token \
  --gateway-token local-dev-token \
  --gateway-bind loopback \
  --skip-channels \
  --skip-search \
  --skip-ui \
  --skip-daemon \
  --skip-health || true

openclaw models list --provider sglang || true
openclaw models status || true
echo

echo "== OpenClaw smoke test =="
openclaw infer model run \
  --local \
  --model "sglang/${MODEL_ID}" \
  --prompt "Reply with exactly: openclaw-sglang-ok" \
  --json
echo

echo "== Run prefix-cache benchmark =="
cd "$PROJECT_ROOT"
python3 bench_sglang_prefix_cache.py \
  --base-url "http://127.0.0.1:${SGLANG_PORT}/v1" \
  --log-file "$SGLANG_LOG" \
  --output-dir "$RESULT_DIR" \
  --timeout 180 \
  --max-tokens 64

echo
echo "== Artifacts =="
echo "SGLang log: $SGLANG_LOG"
echo "Benchmark results: $RESULT_DIR"
find "$RESULT_DIR" -maxdepth 1 -type f -print
echo
echo "== Cache log sample =="
grep -E "#cached-token|cached_tokens|request.finished" "$SGLANG_LOG" | tail -n 40 || true
