#!/bin/bash
#SBATCH --job-name=ws03_hsdp_2x4
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=2
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

CONFIG_PATH="${WORKSHOP_ROOT}/03_distributed/configs/fsdp2_multinode_2x4.yaml"
CKPT_DIR="${CKPT_ROOT}/hsdp_2x4"
mkdir -p "${CKPT_DIR}"

# srun torchrun with c10d rendezvous across 2 nodes × 4 tasks = 8 ranks total.
# automodel's Python launcher handles local rank discovery from torchrun env vars.
srun --kill-on-bad-exit=1 \
    "${WORKSHOP_ROOT}/shared/launch.sh" bash -c "
torchrun \
    --nnodes=${NNODES} \
    --nproc-per-node=${GPUS_PER_NODE} \
    --node-rank=\${SLURM_NODEID} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    /opt/Automodel/nemo_automodel/recipes/llm/train_ft.py \
    --config ${CONFIG_PATH} \
    --dataset.file_pattern='${DATA_ROOT}/fineweb_500M/*.bin' \
    --checkpoint.checkpoint_dir=${CKPT_DIR}
"
