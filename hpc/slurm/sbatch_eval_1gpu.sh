#!/bin/bash
#SBATCH -J drvd_eval
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH -t 08:00:00
#SBATCH --account=mst114553
#SBATCH -o /work/foobarbaz911/vlmr1/logs/drvd_eval_%j.out
#SBATCH -e /work/foobarbaz911/vlmr1/logs/drvd_eval_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[SBATCH] JOBID=${SLURM_JOB_ID}"
echo "[SBATCH] REPO_ROOT=${REPO_ROOT}"

# 1) 啟用你的 uv venv + 環境變數（WORK_DIR / VENV_DIR / HF_HOME ...）
source "${REPO_ROOT}/hpc/bin/activate.sh"

# 2) 你固定的路徑（也可以放到 project.env / project.config）
MODEL_PATH="/work/foobarbaz911/vlmr1/models/VLM-R1-Qwen2.5VL-3B-Math-0305"
DRVD_REPO="/work/foobarbaz911/vlmr1/datasets/DrVD-Bench-repo"
OUT_DIR="/work/foobarbaz911/vlmr1/outputs"
MODEL_TAG="qwen3b"

mkdir -p "${OUT_DIR}"

# 3) 四個任務依序跑（同一張卡，時間要拉長是合理的）
for TASK in independent_qa joint_qa visual_evidence_qa report_generation; do
  echo "[SBATCH] === RUN TASK=${TASK} ==="
  python "${REPO_ROOT}/all_drvd_eval.py" \
    --task "${TASK}" \
    --model-path "${MODEL_PATH}" \
    --drvd-repo "${DRVD_REPO}" \
    --out-dir "${OUT_DIR}" \
    --model-tag "${MODEL_TAG}" \
    --attn-impl sdpa
done

echo "[SBATCH] ALL DONE."