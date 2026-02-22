#!/usr/bin/env bash
set -euo pipefail

die() { echo "[FATAL] $*" >&2; exit 1; }
log() { echo "[INFO]  $*" >&2; }

# 專案根目錄：預設用目前工作目錄，或由外部指定 PROJECT_ROOT
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
