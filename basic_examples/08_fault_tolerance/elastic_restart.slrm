#!/bin/bash
#SBATCH --job-name=ws08_elastic
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

# Torch Elastic demo. Launches training via torchrun with --max-restarts=3,
# then injects a rank-1 crash via chaos_monkey.py. torch elastic should
# respawn the whole worker group and the recipe's restore_from:LATEST picks
# up from the last DCP checkpoint.
#
# Expected: "Max restarts" in torchrun logs, then a second "step 0" after
# restart, which starts from restored optimizer state (loss ≈ loss at crash).

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

CKPT_DIR="${CKPT_ROOT}/ft_elastic"
CONFIG="${WORKSHOP_ROOT}/08_fault_tolerance/configs/tiny_resume.yaml"
rm -rf "${CKPT_DIR}" && mkdir -p "${CKPT_DIR}"

# Build the torchrun command. Use the 4-GPU node as both the rendezvous host
# and the only node group. --max-restarts=3 gives torch elastic a budget for
# worker respawns after crashes.
"${WORKSHOP_ROOT}/shared/launch.sh" bash -c "
torchrun \
    --nnodes=1 \
    --nproc-per-node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --max-restarts=3 \
    --monitor-interval=5 \
    ${WORKSHOP_ROOT}/08_fault_tolerance/chaos_monkey.py \
        --crash-rank=1 --crash-after=45 -- \
        python /opt/Automodel/nemo_automodel/recipes/llm/train_ft.py \
            --config ${CONFIG} \
            --dataset.file_pattern='${DATA_ROOT}/shakespeare/train.bin' \
            --checkpoint.checkpoint_dir=${CKPT_DIR}
"

echo
echo "[elastic] checkpoints landed:"
ls -d "${CKPT_DIR}"/epoch_*_step_* | sort | tail
