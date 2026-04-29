#!/bin/bash
#SBATCH --job-name=ws16_mamba
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

CONFIG="${WORKSHOP_ROOT}/16_mamba/configs/tiny_mamba2_shakespeare.yaml"
CKPT_DIR="${CKPT_ROOT}/tiny_mamba2"

"${WORKSHOP_ROOT}/02_pretrain/pretrain.sh" \
    "${CONFIG}" \
    "${DATA_ROOT}/shakespeare/train.bin" \
    "${CKPT_DIR}" \
    --nproc-per-node=1
