#!/bin/bash
#SBATCH --job-name=ws07_judge
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"
source "${WORKSHOP_ROOT}/shared/lib.sh"

PROMPTS="${PROMPTS:-${RESULTS_ROOT}/prompts/squad_100.jsonl}"
RESPONSES="${RESPONSES:-${RESULTS_ROOT}/outputs/model_under_test.jsonl}"
OUT="${OUT:-${RESULTS_ROOT}/judge/rubric.jsonl}"
JUDGE="${JUDGE:-local:Qwen/Qwen2.5-7B-Instruct}"

# Build the prompts file if missing.
if [[ ! -f "${PROMPTS}" ]]; then
    echo "[judge] prompts file missing → building from SQuAD validation (n=100)"
    "${WORKSHOP_ROOT}/shared/launch.sh" python \
        "${WORKSHOP_ROOT}/07_eval/build_eval_prompts.py" \
        --dataset rajpurkar/squad --n 100 --out "${PROMPTS}"
fi

# Build the responses file if missing (requires a Track B checkpoint).
if [[ ! -f "${RESPONSES}" ]]; then
    CKPT="${CKPT:-$(latest_consolidated_ckpt "${CKPT_ROOT}/trackB_"*)}"
    if [[ -z "${CKPT}" ]]; then
        echo "Set RESPONSES=<jsonl> or CKPT=<ckpt_dir> or run Module 04 Track B first." >&2
        exit 1
    fi
    echo "[judge] responses missing → running batch_infer from ${CKPT}"
    "${WORKSHOP_ROOT}/shared/launch.sh" python \
        "${WORKSHOP_ROOT}/05_inference/batch_infer.py" \
        --ckpt "${CKPT}" --in "${PROMPTS}" --out "${RESPONSES}" \
        --batch-size 8 --use-chat-template
fi

"${WORKSHOP_ROOT}/shared/launch.sh" python \
    "${WORKSHOP_ROOT}/07_eval/llm_judge.py" \
    --prompts "${PROMPTS}" \
    --responses "${RESPONSES}" \
    --judge "${JUDGE}" \
    --out "${OUT}"

echo "[judge] → ${OUT}"
