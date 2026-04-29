#!/bin/bash
# Launch a VLM finetune via the `vlm` CLI domain.
#
# Usage:
#   15_vlm/vlm_finetune.sh <config.yaml> <ckpt_dir> [--nproc-per-node=N] [extra --overrides]

set -euo pipefail

CONFIG="${1:?usage: vlm_finetune.sh <config.yaml> <ckpt_dir> [extra]}"
CKPT_DIR="${2:?ckpt_dir}"
shift 2

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/lib.sh"

CONFIG=$(abs_path "${CONFIG}")
mkdir -p "${CKPT_DIR}"

exec "${WORKSHOP_ROOT}/shared/launch.sh" automodel -c "${CONFIG}" finetune vlm \
    --checkpoint.checkpoint_dir="${CKPT_DIR}" \
    "$@"
