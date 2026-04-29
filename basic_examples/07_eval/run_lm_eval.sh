#!/bin/bash
# Run lm-eval-harness on an HF-format checkpoint, using the writable overlay
# produced by install_overlay.sh.
#
# Usage:
#   07_eval/run_lm_eval.sh <ckpt_dir> [tasks=hellaswag,arc_easy,mmlu] [batch_size=16]
set -euo pipefail

CKPT="${1:?usage: run_lm_eval.sh <ckpt_dir> [tasks] [batch_size]}"
TASKS="${2:-hellaswag,arc_easy}"
BATCH_SIZE="${3:-16}"

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OVERLAY="${OVERLAY:-${WORKSHOP_ROOT}/07_eval/overlay.img}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/kempner_dev/Lab/${USER}/Agent/nemo/runs/results}"
OUT="${RESULTS_ROOT}/lm_eval/$(basename "${CKPT}").json"
mkdir -p "$(dirname "${OUT}")"

OVERLAY="${OVERLAY}" "${WORKSHOP_ROOT}/shared/launch.sh" lm_eval \
    --model hf \
    --model_args "pretrained=${CKPT},dtype=bfloat16,trust_remote_code=true" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUT}" \
    --log_samples

echo "[lm_eval] → ${OUT}"
