#!/usr/bin/env bash
set -

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"euo pipefail
source "$SCRIPT_DIR/common.sh"
source "$SCRIPT_DIR/activate.sh"
source "$SCRIPT_DIR/load_env.sh"

RUN_TAG="${RUN_TAG:-${EXP_NAME:-train}}"
OUT_DIR="$WORK_DIR/outputs/$RUN_TAG"
mkdir -p "$OUT_DIR"

echo "[RUN] Training placeholder. Put your torchrun/deepspeed here."
