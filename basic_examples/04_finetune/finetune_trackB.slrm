#!/bin/bash
#SBATCH --job-name=ws04_trackB_qwen_lora
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

# Choose Qwen3-0.6B (default, ungated) or Llama-3.2-1B (needs HF_TOKEN).
CFG_NAME="${CONFIG:-qwen3_0p6b_lora_squad.yaml}"
CONFIG_PATH="${WORKSHOP_ROOT}/04_finetune/configs/${CFG_NAME}"
CKPT_DIR="${CKPT_ROOT}/trackB_${CFG_NAME%.yaml}"

"${WORKSHOP_ROOT}/04_finetune/finetune.sh" \
    "${CONFIG_PATH}" \
    "${CKPT_DIR}" \
    --nproc-per-node=4
