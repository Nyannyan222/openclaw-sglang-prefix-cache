#!/usr/bin/env bash
# Run OpenClaw + SGLang setup and the prefix-cache benchmark inside a SLURM GPU job.
#
# Submit from the repository root:
#   sbatch --account=<project_id> --time=01:00:00 scripts/slurm_setup_and_benchmark.sh
#
# If your site requires a different partition/account, override at submission:
#   sbatch -p <partition> -A <account> --time=01:00:00 scripts/slurm_setup_and_benchmark.sh

#SBATCH --job-name=oc-sglang-cache
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j-openclaw-sglang.out

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
SGLANG_LOG="$LOG_DIR/sglang_openclaw_${MODEL_LABEL}_${SLURM_JOB_ID:-manual}.log"
RESULT_DIR="$PROJECT_ROOT/benchmark_results/neno5_${MODEL_LABEL}_${SLURM_JOB_ID:-manual}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NODE_VERSION="${NODE_VERSION:-v22.19.0}"
OPENCLAW_PACKAGE="${OPENCLAW_PACKAGE:-openclaw@latest}"
SGLANG_PACKAGE="${SGLANG_PACKAGE:-sglang==0.5.9}"
RESET_VENV="${RESET_VENV:-0}"
MODE="${MODE:-all}"
MANAGED_PYTHON_VERSION="${MANAGED_PYTHON_VERSION:-3.11}"
RUN_OPENCLAW_SMOKE="${RUN_OPENCLAW_SMOKE:-0}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.45}"

case "$MODE" in
  all|setup|run) ;;
  *)
    echo "ERROR: MODE must be one of: all, setup, run"
    exit 1
    ;;
esac

mkdir -p "$RUNTIME_DIR" "$LOG_DIR" "$HF_HOME_DIR" "$NPM_PREFIX" "$NODE_INSTALL_DIR" "$UV_BIN_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$RESULT_DIR"

export PATH="$UV_BIN_DIR:$NODE_INSTALL_DIR/bin:$NPM_PREFIX/bin:$HOME/.local/bin:$PATH"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-false}"

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

node_version_ok() {
  command -v node >/dev/null 2>&1 &&
    command -v npm >/dev/null 2>&1 &&
    node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 22 || (major === 22 && minor >= 19) ? 0 : 1)' >/dev/null 2>&1
}

if ! node_version_ok; then
  try_module_load nodejs node node/22 nodejs/22 node/24 nodejs/24 npm || true
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
  if node_version_ok; then
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

  if ! node_version_ok || [ ! -x "$NODE_INSTALL_DIR/bin/node" ]; then
    if command -v node >/dev/null 2>&1; then
      echo "Existing Node.js is too old for OpenClaw: $(node --version)"
    fi
    echo "Installing local Node.js ${NODE_VERSION} into $NODE_INSTALL_DIR"
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
  node_version_ok
}

ensure_node() {
  if node_version_ok; then
    return 0
  fi
  if [ "$MODE" = "run" ]; then
    echo "ERROR: Node.js 22.19+ is not ready. Run setup first:"
    echo "  sbatch --account=<project_id> scripts/slurm_setup_env.sh"
    exit 1
  fi
  install_local_node
}

ensure_node

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install --user uv
  elif "$PYTHON_BIN" -m ensurepip --user >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install --user uv
  else
    echo "pip is not available; installing uv with the standalone installer into $UV_BIN_DIR"
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_BIN_DIR" sh
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_BIN_DIR" sh
    else
      echo "ERROR: neither curl nor wget is available to install uv."
      return 1
    fi
  fi

  export PATH="$UV_BIN_DIR:$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1
}

python_has_headers() {
  [ -x "$1" ] || command -v "$1" >/dev/null 2>&1 || return 1
  "$1" - <<'PY' >/dev/null 2>&1
from pathlib import Path
import sysconfig

include = sysconfig.get_config_var("INCLUDEPY")
raise SystemExit(0 if include and (Path(include) / "Python.h").exists() else 1)
PY
}

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
    /usr/local/cuda-12.4 \
    /usr/local/cuda-12 \
    /opt/cuda \
    /opt/cuda-12.8 \
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

  try_module_load cuda/12.8 cuda/12.4 cuda/12 cuda cudatoolkit/12.8 cudatoolkit/12.4 cudatoolkit || true
  set_cuda_home_from_nvcc
  if ! cuda_home_ok; then
    set_cuda_home_from_candidates || true
  fi

  if ! cuda_home_ok; then
    echo "ERROR: CUDA toolkit was not found. SGLang/Triton JIT needs CUDA_HOME and nvcc."
    echo "Try checking CUDA modules on the login node:"
    echo "  module avail cuda"
    echo "Then run, for example:"
    echo "  sbatch --account=<project_id> --export=ALL,CUDA_MODULE=<module-name> scripts/slurm_run_benchmark.sh"
    exit 1
  fi

  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
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
    echo "Try checking compiler modules on the login node:"
    echo "  module avail gcc"
    echo "Then run, for example:"
    echo "  sbatch --account=<project_id> --export=ALL,CUDA_MODULE=cuda/12.4,CXX_MODULE=<module-name> scripts/slurm_run_benchmark.sh"
    exit 1
  fi
}

ensure_managed_python() {
  if python_has_headers "$PYTHON_BIN"; then
    echo "Selected Python has headers: $PYTHON_BIN"
    return 0
  fi

  if [ "$MODE" = "run" ]; then
    echo "ERROR: selected Python does not provide Python.h, and run mode will not install Python."
    echo "Run setup first:"
    echo "  sbatch --account=<project_id> scripts/slurm_setup_env.sh"
    exit 1
  fi

  echo "Selected Python lacks Python.h; installing uv-managed Python $MANAGED_PYTHON_VERSION"
  uv python install "$MANAGED_PYTHON_VERSION"
  PYTHON_BIN="$(uv python find "$MANAGED_PYTHON_VERSION")"
  export PYTHON_BIN

  if ! python_has_headers "$PYTHON_BIN"; then
    echo "ERROR: uv-managed Python was installed but Python.h is still missing."
    "$PYTHON_BIN" - <<'PY' || true
import sysconfig
print("INCLUDEPY", sysconfig.get_config_var("INCLUDEPY"))
PY
    exit 1
  fi

  RESET_VENV=1
  export RESET_VENV
  echo "Using uv-managed Python with headers: $PYTHON_BIN"
}

install_openclaw() {
  if command -v openclaw >/dev/null 2>&1 && openclaw --version >/dev/null 2>&1; then
    return 0
  fi

  echo "Installing OpenClaw package: $OPENCLAW_PACKAGE"
  npm install -g "$OPENCLAW_PACKAGE" --prefix "$NPM_PREFIX"
  export PATH="$NPM_PREFIX/bin:$PATH"

  if ! command -v openclaw >/dev/null 2>&1 || ! openclaw --version >/dev/null 2>&1; then
    echo "ERROR: OpenClaw installation finished, but the openclaw command is not runnable."
    echo "Node: $(node --version 2>/dev/null || echo missing)"
    echo "npm: $(npm --version 2>/dev/null || echo missing)"
    return 1
  fi
}

venv_needs_recreate() {
  if [ "$RESET_VENV" = "1" ]; then
    return 0
  fi
  if [ ! -x "$RUNTIME_DIR/.venv/bin/python" ]; then
    return 0
  fi
  if ! python_has_headers "$RUNTIME_DIR/.venv/bin/python"; then
    return 0
  fi
  if "$RUNTIME_DIR/.venv/bin/python" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if sys.version_info >= (3, 9) else 1)
PY
  then
    return 1
  fi
  return 0
}

installed_sglang_matches() {
  [ -x "$RUNTIME_DIR/.venv/bin/python" ] || return 1
  "$RUNTIME_DIR/.venv/bin/python" - "$SGLANG_PACKAGE" <<'PY' >/dev/null 2>&1
import importlib.metadata
import sys

target = sys.argv[1]
if "==" not in target:
    raise SystemExit(0)
_, expected = target.split("==", 1)
actual = importlib.metadata.version("sglang")
raise SystemExit(0 if actual == expected else 1)
PY
}

deep_gemm_absent() {
  [ -x "$RUNTIME_DIR/.venv/bin/python" ] || return 1
  "$RUNTIME_DIR/.venv/bin/python" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("deep_gemm") is None else 1)
PY
}

deep_gemm_module_dir() {
  [ -x "$RUNTIME_DIR/.venv/bin/python" ] || return 1
  "$RUNTIME_DIR/.venv/bin/python" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("deep_gemm")
if spec is None or spec.origin is None:
    raise SystemExit(1)
origin = Path(spec.origin).resolve()
print(origin.parent if origin.name == "__init__.py" else origin)
PY
}

remove_deep_gemm() {
  if deep_gemm_absent; then
    return 0
  fi
  echo "Removing DeepGEMM from the SGLang venv; it requires CUDA_HOME/Python.h on this cluster."
  uv pip uninstall -y deep_gemm deep-gemm || true
  if ! deep_gemm_absent; then
    local module_dir
    local disabled_dir
    module_dir="$(deep_gemm_module_dir)"
    case "$module_dir" in
      "$RUNTIME_DIR"/.venv/*/site-packages/deep_gemm)
        disabled_dir="$RUNTIME_DIR/disabled_python_packages/deep_gemm_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$(dirname "$disabled_dir")"
        echo "Quarantining unmanaged DeepGEMM module: $module_dir -> $disabled_dir"
        mv "$module_dir" "$disabled_dir"
        ;;
      *)
        echo "ERROR: refusing to move unexpected DeepGEMM path: $module_dir"
        return 1
        ;;
    esac
  fi
  deep_gemm_absent
}

reset_sglang_venv() {
  local venv_path="$RUNTIME_DIR/.venv"
  case "$venv_path" in
    "$WORK_ROOT"/runtime/.venv)
      rm -rf "$venv_path"
      ;;
    *)
      echo "ERROR: refusing to remove unexpected venv path: $venv_path"
      exit 1
      ;;
  esac
}

assert_sglang_alive() {
  if ! kill -0 "$SGLANG_PID" >/dev/null 2>&1; then
    echo "ERROR: SGLang process is not running before: $1"
    echo "Last SGLang log lines:"
    tail -n 180 "$SGLANG_LOG" || true
    exit 1
  fi
  if ! curl -fsS "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: SGLang HTTP endpoint is not healthy before: $1"
    echo "Last SGLang log lines:"
    tail -n 180 "$SGLANG_LOG" || true
    exit 1
  fi
}

echo "== Job context =="
date
hostname
whoami
pwd
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "WORK_ROOT=$WORK_ROOT"
echo "MODEL_ID=$MODEL_ID"
echo "OPENCLAW_PACKAGE=$OPENCLAW_PACKAGE"
echo "SGLANG_PACKAGE=$SGLANG_PACKAGE"
echo "MODE=$MODE"
echo "MANAGED_PYTHON_VERSION=$MANAGED_PYTHON_VERSION"
echo "RUN_OPENCLAW_SMOKE=$RUN_OPENCLAW_SMOKE"
echo "SGLANG_MEM_FRACTION_STATIC=$SGLANG_MEM_FRACTION_STATIC"
echo

echo "== GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not available. This job must run on a GPU compute node."
  exit 1
fi
nvidia-smi
echo

echo "== Ensure CUDA toolkit for SGLang/Triton JIT =="
ensure_cuda_home
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version | sed -n '1,4p'
echo

echo "== Ensure C++20 compiler for SGLang JIT =="
ensure_cpp20_toolchain
echo "CC=${CC:-$(command -v gcc || echo unset)}"
echo "CXX=${CXX:-$(command -v g++ || echo unset)}"
"${CXX:-g++}" --version | sed -n '1,3p'
echo

echo "== Tool versions =="
"$PYTHON_BIN" --version
if ! node_version_ok; then
  echo "ERROR: Node.js 22.19+ with npm is required."
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
if [ "$MODE" = "run" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not ready. Run setup first:"
    echo "  sbatch --account=<project_id> scripts/slurm_setup_env.sh"
    exit 1
  fi
else
  install_uv
fi
uv --version
echo

echo "== Ensure Python headers for Triton =="
if [ "$MODE" = "run" ]; then
  if [ ! -x "$RUNTIME_DIR/.venv/bin/python" ] || ! python_has_headers "$RUNTIME_DIR/.venv/bin/python"; then
    echo "ERROR: SGLang venv Python lacks Python.h, which Triton needs at runtime."
    echo "Run setup to rebuild the venv with uv-managed Python:"
    echo "  sbatch --account=<project_id> scripts/slurm_setup_env.sh"
    exit 1
  fi
  echo "SGLang venv Python has headers: $RUNTIME_DIR/.venv/bin/python"
else
  ensure_managed_python
fi
"$PYTHON_BIN" --version
echo

echo "== Install OpenClaw if needed =="
if [ "$MODE" = "run" ]; then
  if ! command -v openclaw >/dev/null 2>&1 || ! openclaw --version >/dev/null 2>&1; then
    echo "ERROR: OpenClaw is not ready. Run setup first:"
    echo "  sbatch --account=<project_id> scripts/slurm_setup_env.sh"
    exit 1
  fi
else
  install_openclaw
fi
openclaw --version
echo

echo "== Create/update SGLang venv =="
cd "$RUNTIME_DIR"
if [ "$MODE" = "run" ]; then
  if venv_needs_recreate || ! installed_sglang_matches; then
    echo "ERROR: SGLang runtime is not ready or does not match $SGLANG_PACKAGE."
    echo "Run setup first:"
    echo "  sbatch --account=<project_id> scripts/slurm_setup_env.sh"
    exit 1
  fi
  if ! deep_gemm_absent; then
    echo "ERROR: deep_gemm is still installed in the SGLang venv and will crash on this cluster."
    echo "Run setup once to remove it:"
    echo "  sbatch --account=<project_id> scripts/slurm_setup_env.sh"
    exit 1
  fi
elif venv_needs_recreate; then
  echo "Creating a fresh SGLang venv with $PYTHON_BIN"
  reset_sglang_venv
  uv venv --python "$PYTHON_BIN" .venv
elif ! installed_sglang_matches; then
  echo "Installed SGLang does not match $SGLANG_PACKAGE; rebuilding venv"
  reset_sglang_venv
  uv venv --python "$PYTHON_BIN" .venv
fi
source .venv/bin/activate
if [ "$MODE" != "run" ]; then
  uv pip install -U "$SGLANG_PACKAGE" hf_transfer
  remove_deep_gemm
fi
python - <<'PY'
import importlib.util
import os
import sys

if importlib.util.find_spec("sglang") is None:
    raise SystemExit("SGLang import check failed")
if importlib.util.find_spec("deep_gemm") is not None:
    raise SystemExit("deep_gemm is still installed; setup should remove it")

print("SGLANG_ENABLE_JIT_DEEPGEMM", os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM"))

import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
else:
    raise SystemExit("PyTorch cannot see CUDA inside the SGLang venv")
PY
echo

if [ "$MODE" = "setup" ]; then
  echo "== Setup complete =="
  echo "Runtime root: $WORK_ROOT"
  echo "Use this command for the benchmark:"
  echo "  sbatch --account=<project_id> scripts/slurm_run_benchmark.sh"
  exit 0
fi

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
assert_sglang_alive "benchmark"

if [ "$RUN_OPENCLAW_SMOKE" = "1" ]; then
  echo "== OpenClaw smoke test =="
  openclaw infer model run \
    --local \
    --model "sglang/${MODEL_ID}" \
    --prompt "Reply with exactly: openclaw-sglang-ok" \
    --json
  echo
  assert_sglang_alive "benchmark after OpenClaw smoke test"
else
  echo "== OpenClaw smoke test skipped =="
  echo "Set RUN_OPENCLAW_SMOKE=1 to run it. The prefix-cache benchmark talks to SGLang directly."
  echo
fi

echo "== Run prefix-cache benchmark =="
cd "$PROJECT_ROOT"
"$RUNTIME_DIR/.venv/bin/python" bench_sglang_prefix_cache.py \
  --base-url "http://127.0.0.1:${SGLANG_PORT}/v1" \
  --model "$MODEL_ID" \
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
