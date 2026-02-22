#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
srun -p normal --gres=gpu:1 --cpus-per-task=4 -t 01:00:00 \
  bash "$PROJECT_ROOT/hpc/bin/run_eval.sh"
