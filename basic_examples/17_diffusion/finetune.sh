#!/bin/bash
# Launch a diffusion finetune job. Uses the bootstrapped main-branch Automodel
# via --bind overlay so TrainDiffusionRecipe is available.
#
# Usage:
#   17_diffusion/finetune.sh <config.yaml> <ckpt_dir> [extra overrides]
#
# Prerequisites:
#   bash 17_diffusion/bootstrap_main.sh    # once, to clone main Automodel

set -euo pipefail

CONFIG="${1:?usage: finetune.sh <config.yaml> <ckpt_dir> [extra overrides]}"
CKPT_DIR="${2:?ckpt_dir (replaces PATH_TO_YOUR_CKPT_DIR in the config)}"
shift 2

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/lib.sh"

MAIN_CLONE="${WORKSHOP_ROOT}/17_diffusion/_automodel_main"
if [[ ! -d "${MAIN_CLONE}" ]]; then
    echo "[diffusion] ${MAIN_CLONE} not found — run 17_diffusion/bootstrap_main.sh first." >&2
    exit 1
fi

CONFIG=$(abs_path "${CONFIG}")
mkdir -p "${CKPT_DIR}"

# Shadow-mount main Automodel over /opt/Automodel so the diffusion recipe is visible.
EXTRA_BINDS="${MAIN_CLONE}:/opt/Automodel" \
exec "${WORKSHOP_ROOT}/shared/launch.sh" \
    python /opt/Automodel/examples/diffusion/finetune/finetune.py \
    --config "${CONFIG}" \
    --checkpoint.checkpoint_dir="${CKPT_DIR}" \
    "$@"
