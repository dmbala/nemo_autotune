#!/bin/bash
# One-time setup: build a writable Singularity overlay image and pip-install
# lm-eval-harness into it. Run on a login node (compute nodes have no internet).
#
# Re-runs are idempotent: existing overlay won't be overwritten.
#
# Usage:
#   07_eval/install_overlay.sh            # → 07_eval/overlay.img (4 GB)

set -euo pipefail

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OVERLAY="${WORKSHOP_ROOT}/07_eval/overlay.img"

if [[ ! -f "${OVERLAY}" ]]; then
    "${WORKSHOP_ROOT}/shared/user_overlay.sh" "${OVERLAY}" 4096
fi

# Install inside the container, writing to the overlay.
OVERLAY="${OVERLAY}" "${WORKSHOP_ROOT}/shared/launch.sh" bash -lc "
pip install --no-cache-dir 'lm-eval[hf]==0.4.*' antlr4-python3-runtime==4.11 && \
    python -c 'import lm_eval; print(\"lm_eval ok:\", lm_eval.__version__)'
"

echo "[overlay] lm-eval installed into ${OVERLAY}"
echo "[overlay] use by passing OVERLAY=${OVERLAY} to shared/launch.sh"
