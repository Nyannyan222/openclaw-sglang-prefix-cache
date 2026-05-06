#!/usr/bin/env bash
# Submit one benchmark job per model. Run this from the login node after setup succeeds.
#
# Usage:
#   bash scripts/slurm_run_model_matrix.sh
#   ACCOUNT=MST114180 MODELS="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct" bash scripts/slurm_run_model_matrix.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-MST114180}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"
PARTITION="${PARTITION:-dev}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/slurm_run_benchmark.sh}"

DEFAULT_MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
)

if [ -n "${MODELS:-}" ]; then
  # shellcheck disable=SC2206
  MODEL_LIST=($MODELS)
else
  MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

echo "Submitting ${#MODEL_LIST[@]} model benchmark job(s)"
echo "ACCOUNT=$ACCOUNT"
echo "PARTITION=$PARTITION"
echo "TIME_LIMIT=$TIME_LIMIT"
echo

for model_id in "${MODEL_LIST[@]}"; do
  model_label="${model_id//\//__}"
  echo "Submitting model: $model_id"
  sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIME_LIMIT" \
    --job-name="oc-sglang-${model_label:0:24}" \
    --export=ALL,MODEL_ID="$model_id",MODEL_LABEL="$model_label" \
    "$RUN_SCRIPT"
done
