#!/bin/bash
#SBATCH --job-name=ws02_pretrain_124m
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

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

"${WORKSHOP_ROOT}/02_pretrain/pretrain.sh" \
    "${WORKSHOP_ROOT}/02_pretrain/configs/gpt2_124m_fineweb.yaml" \
    "${DATA_ROOT}/fineweb_500M/*.bin" \
    "${CKPT_ROOT}/gpt2_124m" \
    --nproc-per-node=4
