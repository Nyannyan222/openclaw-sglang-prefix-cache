#!/usr/bin/env bash
# Run SGLang and the R1/R2/R3 prefix-cache benchmark using an existing setup.
#
# Submit from the repository root after scripts/slurm_setup_env.sh succeeds:
#   sbatch --account=<project_id> scripts/slurm_run_benchmark.sh

#SBATCH --job-name=oc-sglang-run
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j-openclaw-sglang-run.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
MODE=run bash scripts/slurm_setup_and_benchmark.sh
