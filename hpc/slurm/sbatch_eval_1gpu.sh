#!/bin/bash
#SBATCH -J eval
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o /work/%u/vlmr1/logs/%x_%j.out
#SBATCH -e /work/%u/vlmr1/logs/%x_%j.err

set -euo pipefail

# 讓你可以在 sbatch 時用 --export 覆蓋
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"

# 直接呼叫專案內的 hpc/bin/run_eval.sh
bash "$PROJECT_ROOT/hpc/bin/run_eval.sh"
