#!/bin/bash
#SBATCH --job-name=ws03_fsdp2_dp4
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

# Choose which config to run (override via env: CONFIG=ckpt_async_consolidated.yaml sbatch ...).
CFG_NAME="${CONFIG:-fsdp2_dp4.yaml}"
CONFIG_PATH="${WORKSHOP_ROOT}/03_distributed/configs/${CFG_NAME}"
CKPT_DIR="${CKPT_ROOT}/fsdp2_${CFG_NAME%.yaml}"
mkdir -p "${CKPT_DIR}"

"${WORKSHOP_ROOT}/02_pretrain/pretrain.sh" \
    "${CONFIG_PATH}" \
    "${DATA_ROOT}/fineweb_500M/*.bin" \
    "${CKPT_DIR}" \
    --nproc-per-node=4
