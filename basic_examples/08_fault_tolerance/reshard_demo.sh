#!/bin/bash
# DCP reshardable-load demo. Save a checkpoint with FSDP2 dp_size=4, then
# resume with dp_size=2. torch.distributed.checkpoint's load path does the
# redistribution for us — the saved model shards are stitched on load and
# re-sharded to the new mesh.
#
# Usage:
#   08_fault_tolerance/reshard_demo.sh <train.bin_glob> <ckpt_dir>
#
# Expects an active 4-GPU Slurm allocation for phase 1 and at least 2 for phase 2.
# Inside a single sbatch with --gres=gpu:4 we reuse the allocation for both phases.

set -euo pipefail

TRAIN_BIN="${1:?usage: reshard_demo.sh <train.bin_glob> <ckpt_dir>}"
CKPT_DIR="${2:?ckpt_dir}"

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${WORKSHOP_ROOT}/08_fault_tolerance/configs/tiny_resume.yaml"

rm -rf "${CKPT_DIR}"
mkdir -p "${CKPT_DIR}"

echo
echo "=== PHASE 1: save at dp=4 (4 GPUs) ==="
echo
"${WORKSHOP_ROOT}/shared/launch.sh" automodel -c "${CONFIG}" pretrain llm \
    --dataset.file_pattern="${TRAIN_BIN}" \
    --checkpoint.checkpoint_dir="${CKPT_DIR}" \
    --step_scheduler.max_steps=20 \
    --step_scheduler.ckpt_every_steps=20 \
    --distributed.dp_size=4 \
    --nproc-per-node=4

LATEST=$(readlink -f "${CKPT_DIR}/LATEST")
echo "[reshard] saved: ${LATEST}"

echo
echo "=== PHASE 2: load same checkpoint at dp=2 (2 GPUs) ==="
echo
# Explicit restore_from path; pretend the checkpoint exists from a prior run.
"${WORKSHOP_ROOT}/shared/launch.sh" automodel -c "${CONFIG}" pretrain llm \
    --dataset.file_pattern="${TRAIN_BIN}" \
    --checkpoint.checkpoint_dir="${CKPT_DIR}" \
    --checkpoint.restore_from="$(basename "${LATEST}")" \
    --step_scheduler.max_steps=30 \
    --step_scheduler.ckpt_every_steps=10 \
    --distributed.dp_size=2 \
    --nproc-per-node=2

echo "[reshard] step count after resume:"
ls -d "${CKPT_DIR}"/epoch_*_step_* | sort | tail -3
