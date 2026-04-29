#!/bin/bash
# Launch an Automodel finetune job inside the container.
#
# Usage:
#   04_finetune/finetune.sh <config.yaml> <ckpt_dir> [--model.pretrained_model_name_or_path=<ckpt>] [other --overrides]
#
# Track A (continue from our pretrained nanoGPT):
#   04_finetune/finetune.sh configs/tiny_sft_squad.yaml $CKPT_ROOT/tiny_sft \
#     --model.pretrained_model_name_or_path=$CKPT_ROOT/tiny_gpt2/epoch_0_step_499/model/consolidated \
#     --nproc-per-node=1
#
# Track B (download an HF model):
#   04_finetune/finetune.sh configs/qwen3_0p6b_lora_squad.yaml $CKPT_ROOT/qwen3_lora \
#     --nproc-per-node=4
set -euo pipefail

CONFIG="${1:?usage: finetune.sh <config.yaml> <ckpt_dir> [extra overrides]}"
CKPT_DIR="${2:?ckpt_dir}"
shift 2

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/lib.sh"

CONFIG=$(abs_path "${CONFIG}")
mkdir -p "${CKPT_DIR}"

exec "${WORKSHOP_ROOT}/shared/launch.sh" automodel -c "${CONFIG}" finetune llm \
    --checkpoint.checkpoint_dir="${CKPT_DIR}" \
    "$@"
