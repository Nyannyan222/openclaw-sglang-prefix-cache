#!/usr/bin/env bash
# Lightweight login-node check. This does not run GPU/server workload.

set -euo pipefail

echo "== Host =="
hostname
whoami
pwd
echo

echo "== Git repo =="
git rev-parse --show-toplevel
git status --short --branch
echo

echo "== Tools =="
python3 --version || true
node --version || true
npm --version || true
command -v uv && uv --version || true
command -v openclaw && openclaw --version || true
command -v sbatch && sbatch --version || true
echo

echo "== Recommended next command =="
echo "sbatch --account=<project_id> scripts/slurm_setup_env.sh"
echo "After setup succeeds:"
echo "sbatch --account=<project_id> scripts/slurm_run_benchmark.sh"
