#!/bin/bash
#SBATCH --job-name=ws17_diff_finetune
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Default: LoRA-finetune Wan2.1 T2V with flow matching.
# Override via env:
#   CONFIG=finetune_flux_lora.yaml   # Flux-T2I LoRA variant
#   CONFIG=finetune_wan2_1_t2v_multinode.yaml  # multinode (set --nodes too)
#
# Prerequisite (run once): bash 17_diffusion/bootstrap_main.sh

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

CFG_NAME="${CONFIG:-finetune_wan2_1_t2v_lora.yaml}"
CONFIG_PATH="${WORKSHOP_ROOT}/17_diffusion/configs/${CFG_NAME}"
CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/diffusion_${CFG_NAME%.yaml}}"

"${WORKSHOP_ROOT}/17_diffusion/finetune.sh" "${CONFIG_PATH}" "${CKPT_DIR}"
