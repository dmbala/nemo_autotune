#!/bin/bash
# Build a writable Singularity overlay image so Module 07 (eval) can pip-install
# packages (lm-eval) on top of the read-only SIF without rebuilding the container.
#
# Usage:
#   shared/user_overlay.sh <overlay.img> [size_mb]
#
# Example:
#   shared/user_overlay.sh 07_eval/overlay.img 4096

set -euo pipefail

OVERLAY_PATH="${1:?usage: user_overlay.sh <path> [size_mb]}"
SIZE_MB="${2:-4096}"

if [[ -e "${OVERLAY_PATH}" ]]; then
    echo "[user_overlay] ${OVERLAY_PATH} already exists; skipping create." >&2
    exit 0
fi

mkdir -p "$(dirname "${OVERLAY_PATH}")"
singularity overlay create --size "${SIZE_MB}" "${OVERLAY_PATH}"
echo "[user_overlay] created ${OVERLAY_PATH} (${SIZE_MB} MB)"
