#!/usr/bin/env bash
# Run OpenClaw + SGLang setup and the prefix-cache benchmark inside a SLURM GPU job.
#
# Submit from the repository root:
#   sbatch scripts/slurm_setup_and_benchmark.sh
#
# If your site requires a different partition/account, override at submission:
#   sbatch -p <partition> -A <account> scripts/slurm_setup_and_benchmark.sh

#SBATCH --job-name=oc-sglang-cache
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j-openclaw-sglang.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORK_ROOT="${WORK_ROOT:-/work/$USER/openclaw-sglang}"
RUNTIME_DIR="$WORK_ROOT/runtime"
LOG_DIR="$WORK_ROOT/logs"
HF_HOME_DIR="$WORK_ROOT/huggingface"
NPM_PREFIX="$WORK_ROOT/npm"
SGLANG_LOG="$LOG_DIR/sglang_openclaw_${SLURM_JOB_ID:-manual}.log"
RESULT_DIR="$PROJECT_ROOT/benchmark_results/neno5_${SLURM_JOB_ID:-manual}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
SGLANG_PORT="${SGLANG_PORT:-30000}"

mkdir -p "$RUNTIME_DIR" "$LOG_DIR" "$HF_HOME_DIR" "$NPM_PREFIX" "$RESULT_DIR"

export PATH="$NPM_PREFIX/bin:$HOME/.local/bin:$PATH"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1

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
python3 --version
node --version
npm --version
echo

echo "== Install uv if needed =="
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user uv
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
  uv venv --python 3.12 .venv
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

