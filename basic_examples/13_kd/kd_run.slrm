#!/bin/bash
#SBATCH --job-name=ws13_kd
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

# Choose config: full-finetune KD (default) or LoRA-KD via CONFIG env.
CFG_NAME="${CONFIG:-kd_qwen3_student0p6b_teacher1p7b.yaml}"
CONFIG_PATH="${WORKSHOP_ROOT}/13_kd/configs/${CFG_NAME}"
CKPT_DIR="${CKPT_ROOT}/kd_${CFG_NAME%.yaml}"

"${WORKSHOP_ROOT}/13_kd/kd.sh" \
    "${CONFIG_PATH}" \
    "${CKPT_DIR}" \
    --nproc-per-node=4
