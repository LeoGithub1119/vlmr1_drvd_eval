#!/usr/bin/env bash
set -

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"euo pipefail
source "$SCRIPT_DIR/common.sh"
source "$SCRIPT_DIR/load_env.sh"

# 要下載什麼由 project.config 控制
# MODEL_ID, DATASET_ID, DATASET_TYPE=dataset|model

mkdir -p "$WORK_DIR/datasets" "$WORK_DIR/models"

if [ -n "${DATASET_ID:-}" ]; then
  echo "[DL] dataset: $DATASET_ID"
  huggingface-cli download "$DATASET_ID" --repo-type dataset \
    --local-dir "$WORK_DIR/datasets/$(basename "$DATASET_ID")"
fi

if [ -n "${MODEL_ID:-}" ]; then
  echo "[DL] model: $MODEL_ID"
  huggingface-cli download "$MODEL_ID" \
    --local-dir "$WORK_DIR/models/$(basename "$MODEL_ID")"
fi
