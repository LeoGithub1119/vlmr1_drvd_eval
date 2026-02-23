#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
source "$SCRIPT_DIR/activate.sh"
source "$SCRIPT_DIR/load_env.sh"

# === 固定路徑 ===
MODEL_PATH="/work/foobarbaz911/vlmr1/models/VLM-R1-Qwen2.5VL-3B-Math-0305"
DRVD_ROOT="/work/foobarbaz911/vlmr1/datasets/DrVD-Bench-repo"
OUT_ROOT="/work/foobarbaz911/vlmr1/outputs"
LOG_ROOT="/work/foobarbaz911/vlmr1/logs"

mkdir -p "$OUT_ROOT"
mkdir -p "$LOG_ROOT"

echo "[RUN] JOBID=$SLURM_JOB_ID"
echo "[RUN] MODEL_PATH=$MODEL_PATH"
echo "[RUN] DRVD_ROOT=$DRVD_ROOT"
echo "[RUN] OUT_ROOT=$OUT_ROOT"

TASKS=("independent" "joint" "visual_evidence" "report_generation")

for task in "${TASKS[@]}"; do
    echo "======================================"
    echo "[RUN] TASK=$task"
    echo "======================================"

    python drvd_local_eval.py \
        --model "$MODEL_PATH" \
        --drvd-root "$DRVD_ROOT" \
        --out-dir "$OUT_ROOT" \
        --tasks "$task" \
        --model-tag qwen3b \
        --save-raw

done

echo "[DONE] All tasks finished."