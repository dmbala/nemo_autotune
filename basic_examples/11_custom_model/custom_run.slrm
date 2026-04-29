#!/bin/bash
#SBATCH --job-name=ws11_custom_model
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

# Make custom_model.py importable inside the container.
export PYTHONPATH="${WORKSHOP_ROOT}/11_custom_model:${PYTHONPATH:-}"

CONFIG="${WORKSHOP_ROOT}/11_custom_model/configs/custom_rope_shakespeare.yaml"
CKPT_DIR="${CKPT_ROOT}/custom_rope_gpt"

"${WORKSHOP_ROOT}/02_pretrain/pretrain.sh" \
    "${CONFIG}" \
    "${DATA_ROOT}/shakespeare/train.bin" \
    "${CKPT_DIR}" \
    --nproc-per-node=1
