#!/bin/bash
# Launch a knowledge-distillation run. Thin wrapper around the automodel CLI.
#
# Usage:
#   13_kd/kd.sh <config.yaml> <ckpt_dir> [extra --dotted overrides]
#
# Example:
#   13_kd/kd.sh configs/kd_qwen3_student0p6b_teacher1p7b.yaml \
#     $CKPT_ROOT/kd_qwen3 --nproc-per-node=4

set -euo pipefail

CONFIG="${1:?usage: kd.sh <config.yaml> <ckpt_dir> [overrides]}"
CKPT_DIR="${2:?ckpt_dir}"
shift 2

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/lib.sh"

CONFIG=$(abs_path "${CONFIG}")
mkdir -p "${CKPT_DIR}"

exec "${WORKSHOP_ROOT}/shared/launch.sh" automodel -c "${CONFIG}" kd llm \
    --checkpoint.checkpoint_dir="${CKPT_DIR}" \
    "$@"
