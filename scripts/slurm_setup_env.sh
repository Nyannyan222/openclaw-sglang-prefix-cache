#!/usr/bin/env bash
# One-time environment setup for OpenClaw + SGLang on a GPU compute node.
#
# Submit from the repository root:
#   sbatch --account=<project_id> scripts/slurm_setup_env.sh

#SBATCH --job-name=oc-sglang-setup
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j-openclaw-sglang-setup.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
MODE=setup bash scripts/slurm_setup_and_benchmark.sh
