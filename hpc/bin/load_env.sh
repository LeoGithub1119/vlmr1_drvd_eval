#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# common.sh 應該會設定 PROJECT_ROOT；若沒有就用 SCRIPT_DIR 往上推
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"


ENV_FILE="$PROJECT_ROOT/project.env"
CFG_FILE="$PROJECT_ROOT/project.config"

[ -f "$ENV_FILE" ] || die "ENV_FILE not found: $ENV_FILE"
[ -f "$CFG_FILE" ] || die "CFG_FILE not found: $CFG_FILE"

# shellcheck disable=SC1090
source "$ENV_FILE"
# shellcheck disable=SC1090
source "$CFG_FILE"

log "Loaded ENV_FILE=$ENV_FILE"
log "Loaded CFG_FILE=$CFG_FILE"
