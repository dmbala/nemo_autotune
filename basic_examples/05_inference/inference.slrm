#!/bin/bash
#SBATCH --job-name=ws05_infer
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
source "${WORKSHOP_ROOT}/shared/lib.sh"

CKPT="${CKPT:-$(latest_consolidated_ckpt "${CKPT_ROOT}/tiny_sft")}"
if [[ -z "${CKPT}" ]]; then
    echo "No consolidated checkpoint found. Pass CKPT=<path> or run Module 02/04 first."
    exit 1
fi
echo "[infer] ckpt=${CKPT}"

# We need a tokenizer. Pretrain-from-scratch ckpts lack one — fall back to gpt2.
TOKENIZER=""
if [[ ! -f "${CKPT}/tokenizer_config.json" && ! -f "${CKPT}/tokenizer.json" ]]; then
    TOKENIZER="--tokenizer gpt2"
fi

"${WORKSHOP_ROOT}/shared/launch.sh" python \
    "${WORKSHOP_ROOT}/05_inference/generate.py" \
    --ckpt "${CKPT}" \
    ${TOKENIZER} \
    --prompt "Context: The Nile is the longest river in Africa. Question: Which river is the longest river in Africa?" \
    --max-new-tokens 80
