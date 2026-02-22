#!/usr/bin/env bash
set -

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"euo pipefail
source "$SCRIPT_DIR/common.sh"
source "$SCRIPT_DIR/activate.sh"
source "$SCRIPT_DIR/load_env.sh"

# 預設輸出檔
RUN_TAG="${RUN_TAG:-${EXP_NAME:-eval}}"
OUT_DIR="$WORK_DIR/outputs/$RUN_TAG"
mkdir -p "$OUT_DIR"

echo "[RUN] EXP_NAME=${EXP_NAME:-}"
echo "[RUN] MODEL_ID=${MODEL_ID:-}"
echo "[RUN] DATASET_ID=${DATASET_ID:-}"
echo "[RUN] OUT_DIR=$OUT_DIR"

# ====== 你之後會把這段改成 VLM-R1 真正的 eval 入口 ======
# 先放一個 smoke 版本，確認環境OK
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" | tee "$OUT_DIR/smoke.txt"

# TODO: 之後替換成類似
# python src/eval/test_rec_r1.py --model_path ... --data_dir ... --output ...
