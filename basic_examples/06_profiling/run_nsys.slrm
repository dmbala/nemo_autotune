#!/bin/bash
#SBATCH --job-name=ws06_nsys
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

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

TRACE_DIR="${RESULTS_ROOT}/nsys"
mkdir -p "${TRACE_DIR}"
CONFIG_PATH="${WORKSHOP_ROOT}/06_profiling/configs/nsys_gpt2_124m.yaml"

# The benchmark recipe calls cudaProfilerStart/Stop around the nsys_start..nsys_end window.
# -c cudaProfilerApi lets nsys honor those markers.
"${WORKSHOP_ROOT}/shared/launch.sh" bash -c "
nsys profile \
    -o ${TRACE_DIR}/trace \
    -c cudaProfilerApi \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --sample=none \
    --force-overwrite true \
    automodel -c ${CONFIG_PATH} benchmark llm --nproc-per-node=4
"

echo "[nsys] trace written to ${TRACE_DIR}/trace.nsys-rep"
echo "[nsys] summarize:  nsys stats ${TRACE_DIR}/trace.nsys-rep | head -40"
