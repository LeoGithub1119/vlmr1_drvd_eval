#!/usr/bin/env bash
# shellcheck shell=bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 必須被 source
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: Please source this script:"
  echo "  source ${BASH_SOURCE[0]}"
  exit 1
fi

# ---- 保存使用者原本 shell 狀態 ----
__old_opts="$(set +o)"
__old_trap_err="$(trap -p ERR || true)"

# ---- activation 期間使用 strict mode ----
set -euo pipefail

# -----------------------------
# Activation start
# -----------------------------

# common helpers
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

# load project env
# shellcheck disable=SC1091
source "$SCRIPT_DIR/load_env.sh"

# modules（避免 set -u 炸 Lmod）
set +u
module purge || true
[ -n "${CUDA_MODULE:-}" ] && module load "${CUDA_MODULE}"
[ -n "${MINICONDA_MODULE:-}" ] && module load "${MINICONDA_MODULE}"
set -u

echo "-----------------"
echo "loading CUDA ${CUDA_VERSION:-}"
echo "-----------------"
echo

# venv
[ -d "${VENV_DIR:-}" ] || {
  echo "VENV_DIR not found: ${VENV_DIR:-<empty>}"
  eval "$__old_opts"
  return 1
}

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# cache dirs
mkdir -p "$WORK_DIR"/{hf_cache,xdg_cache,datasets,models,outputs,logs} || true

export HF_HOME="$WORK_DIR/hf_cache"
export HF_HUB_CACHE="$WORK_DIR/hf_cache"
export HF_DATASETS_CACHE="$WORK_DIR/hf_cache"
export TRANSFORMERS_CACHE="$WORK_DIR/hf_cache"
export XDG_CACHE_HOME="$WORK_DIR/xdg_cache"
export TOKENIZERS_PARALLELISM=false

echo "[INFO]  Loaded ENV_FILE=${ENV_FILE:-<unset>}"
echo "[INFO]  Loaded CFG_FILE=${CFG_FILE:-<unset>}"
echo "[ENV] WORK_DIR=$WORK_DIR"
echo "[ENV] VENV_DIR=$VENV_DIR"
python -V

# -----------------------------
# 還原使用者 shell 狀態
# -----------------------------
eval "$__old_opts"
eval "$__old_trap_err"

unset __old_opts __old_trap_err