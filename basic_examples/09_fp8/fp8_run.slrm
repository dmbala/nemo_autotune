#!/bin/bash
#SBATCH --job-name=ws09_fp8
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

CONFIG_PATH="${WORKSHOP_ROOT}/09_fp8/configs/fp8_qwen3_0p6b_lora.yaml"
CKPT_DIR="${CKPT_ROOT}/fp8_qwen3_lora"

"${WORKSHOP_ROOT}/04_finetune/finetune.sh" \
    "${CONFIG_PATH}" \
    "${CKPT_DIR}" \
    --nproc-per-node=4
