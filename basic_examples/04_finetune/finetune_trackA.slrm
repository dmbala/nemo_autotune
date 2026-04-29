#!/bin/bash
#SBATCH --job-name=ws04_trackA_sft
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
source "${WORKSHOP_ROOT}/shared/lib.sh"

PRETRAINED_DIR=$(latest_consolidated_ckpt "${CKPT_ROOT}/tiny_gpt2")
if [[ -z "${PRETRAINED_DIR}" ]]; then
    echo "No consolidated checkpoint found under ${CKPT_ROOT}/tiny_gpt2. Did Module 02 pretrain finish?"
    exit 1
fi
echo "[trackA] resume from ${PRETRAINED_DIR}"

"${WORKSHOP_ROOT}/04_finetune/finetune.sh" \
    "${WORKSHOP_ROOT}/04_finetune/configs/tiny_sft_squad.yaml" \
    "${CKPT_ROOT}/tiny_sft" \
    --model.pretrained_model_name_or_path="${PRETRAINED_DIR}" \
    --nproc-per-node=1
