#!/bin/bash
#SBATCH --job-name=ws07_lm_eval
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"
source "${WORKSHOP_ROOT}/shared/lib.sh"

CKPT="${CKPT:-$(latest_consolidated_ckpt "${CKPT_ROOT}/trackB_"*)}"
if [[ -z "${CKPT}" ]]; then
    echo "No consolidated checkpoint found. Pass CKPT=<path> or run Module 04 Track B first."
    exit 1
fi

TASKS="${TASKS:-hellaswag,arc_easy}"
BATCH_SIZE="${BATCH_SIZE:-16}"

OVERLAY="${WORKSHOP_ROOT}/07_eval/overlay.img" \
"${WORKSHOP_ROOT}/07_eval/run_lm_eval.sh" "${CKPT}" "${TASKS}" "${BATCH_SIZE}"
