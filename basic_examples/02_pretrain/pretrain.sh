#!/bin/bash
# Launch a NeMo-Automodel pretraining run inside the container.
#
# Usage:
#   02_pretrain/pretrain.sh <config.yaml> <train.bin> <ckpt_dir> [--nproc-per-node=N] [extra --dotted.overrides]
#
# Examples:
#   02_pretrain/pretrain.sh configs/tiny_gpt2_shakespeare.yaml \
#     $DATA_ROOT/shakespeare/train.bin $CKPT_ROOT/tiny_gpt2 --nproc-per-node=1
#
#   02_pretrain/pretrain.sh configs/gpt2_124m_fineweb.yaml \
#     "$DATA_ROOT/fineweb_500M/*.bin" $CKPT_ROOT/gpt2_124m --nproc-per-node=4

set -euo pipefail

CONFIG="${1:?usage: pretrain.sh <config.yaml> <train.bin_or_glob> <ckpt_dir> [extra]}"
TRAIN_BIN="${2:?train.bin path or glob}"
CKPT_DIR="${3:?ckpt_dir}"
shift 3

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/lib.sh"

CONFIG=$(abs_path "${CONFIG}")
mkdir -p "${CKPT_DIR}"

exec "${WORKSHOP_ROOT}/shared/launch.sh" automodel -c "${CONFIG}" pretrain llm \
    --dataset.file_pattern="${TRAIN_BIN}" \
    --checkpoint.checkpoint_dir="${CKPT_DIR}" \
    "$@"
