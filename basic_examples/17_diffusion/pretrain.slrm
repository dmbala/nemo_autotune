#!/bin/bash
#SBATCH --job-name=ws17_diff_pretrain
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Diffusion / flow-matching pretrain. The upstream configs assume dp_size=8 on
# one node. For real pretrain runs, scale up nodes + batch via the multinode
# variant under finetune/ (the recipe is shared).
#
# Override via env:
#   CONFIG=pretrain_flux_flow.yaml            # Flux-T2I (image)
#   CONFIG=pretrain_wan2_1_t2v_flow.yaml      # Wan2.1 T2V (video, default)

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

CFG_NAME="${CONFIG:-pretrain_wan2_1_t2v_flow.yaml}"
CONFIG_PATH="${WORKSHOP_ROOT}/17_diffusion/configs/${CFG_NAME}"
CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/diffusion_${CFG_NAME%.yaml}}"

"${WORKSHOP_ROOT}/17_diffusion/pretrain.sh" "${CONFIG_PATH}" "${CKPT_DIR}"
