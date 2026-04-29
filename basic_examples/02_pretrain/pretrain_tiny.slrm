#!/bin/bash
#SBATCH --job-name=ws02_pretrain_tiny
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

"${WORKSHOP_ROOT}/02_pretrain/pretrain.sh" \
    "${WORKSHOP_ROOT}/02_pretrain/configs/tiny_gpt2_shakespeare.yaml" \
    "${DATA_ROOT}/shakespeare/train.bin" \
    "${CKPT_ROOT}/tiny_gpt2" \
    --nproc-per-node=1
