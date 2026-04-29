#!/bin/bash
#SBATCH --job-name=ws17_diffusion
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

# Distributed video generation with Wan2.2-T2V-A14B via NeMoAutoDiffusionPipeline.
# 14B DiT is sharded TP=4 across one H100 node. ~5-10 min for a 3-second 480x848 clip.
#
# Override via env:
#   PROMPT="..."           — text prompt (safe to contain quotes/shell metacharacters)
#   MODEL=...              — any diffusers video pipeline
#   NUM_STEPS=20           — denoising steps
#   HEIGHT/WIDTH/FRAMES    — output geometry
#   OUTPUT=/path/out.mp4   — output path (defaults to $RESULTS_ROOT/diffusion/<jobid>.mp4)

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

export WS_MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
export WS_PROMPT="${PROMPT:-A cinematic aerial shot of a snow-capped mountain range at sunrise.}"
export WS_NUM_STEPS="${NUM_STEPS:-20}"
export WS_HEIGHT="${HEIGHT:-480}"
export WS_WIDTH="${WIDTH:-848}"
export WS_FRAMES="${FRAMES:-81}"
OUT_DIR="${RESULTS_ROOT}/diffusion"
mkdir -p "${OUT_DIR}"
export WS_OUTPUT="${OUTPUT:-${OUT_DIR}/${SLURM_JOB_ID:-local}.mp4}"

# Pass torchrun flags as a pre-formed arg array so we don't invoke `bash -c` with
# string-interpolated user input (previously an injection hazard on PROMPT).
"${WORKSHOP_ROOT}/shared/launch.sh" torchrun \
    --nproc-per-node="${GPUS_PER_NODE}" \
    --nnodes="${NNODES}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${WORKSHOP_ROOT}/17_diffusion/generate.py" \
    --model "${WS_MODEL}" \
    --prompt "${WS_PROMPT}" \
    --num-inference-steps "${WS_NUM_STEPS}" \
    --height "${WS_HEIGHT}" --width "${WS_WIDTH}" --num-frames "${WS_FRAMES}" \
    --tp-size "${GPUS_PER_NODE}" \
    --output "${WS_OUTPUT}"

echo "[diffusion] video → ${WS_OUTPUT}"
