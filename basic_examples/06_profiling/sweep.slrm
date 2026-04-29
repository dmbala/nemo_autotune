#!/bin/bash
#SBATCH --job-name=ws06_sweep
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

SWEEP_DIR="${RESULTS_ROOT}/sweep"

"${WORKSHOP_ROOT}/shared/launch.sh" python \
    "${WORKSHOP_ROOT}/06_profiling/sweep.py" \
    --base "${WORKSHOP_ROOT}/06_profiling/configs/benchmark_base.yaml" \
    --results "${SWEEP_DIR}" \
    --batch-sizes 1,2,4,8 \
    --seq-lens 512,1024 \
    --dp-tp "4x1" \
    --nproc-per-node 4 \
    --max-steps 20

"${WORKSHOP_ROOT}/shared/launch.sh" python \
    "${WORKSHOP_ROOT}/06_profiling/report.py" --results "${SWEEP_DIR}"
