#!/bin/bash
#SBATCH --job-name=cb_pair_elem
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=00:30:00

# Single array element for the all-pairs NCCL matrix. Reads its assigned pair
# from ${PAIRS_FILE} at line ${SLURM_ARRAY_TASK_ID}+1 and runs a 2-node
# all_reduce between those two specific nodes.
#
# Launched by pair_matrix.sh. Don't sbatch this directly — the env vars it
# needs are exported by that driver.

set -euo pipefail

: "${PAIRS_FILE:?PAIRS_FILE unset — submit via pair_matrix.sh, not directly}"
: "${PAIR_OUT_DIR:?PAIR_OUT_DIR unset}"

CLUSTER_BENCH_ROOT="${CLUSTER_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${CLUSTER_BENCH_ROOT}/shared/slurm_common.sh"

idx=$((SLURM_ARRAY_TASK_ID + 1))
read -r NODE_A NODE_B < <(sed -n "${idx}p" "${PAIRS_FILE}")
if [[ -z "${NODE_A:-}" || -z "${NODE_B:-}" ]]; then
    echo "[pair_elem] could not read pair at line ${idx} of ${PAIRS_FILE}" >&2
    exit 1
fi

# Double-check Slurm actually gave us those nodes (protects against the job
# getting scheduled to a different pair if --nodelist isn't honored).
actual_nodes="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sort | tr '\n' ' ')"
echo "[pair_elem] array=${SLURM_ARRAY_TASK_ID} pair=(${NODE_A} ${NODE_B}) actual=${actual_nodes}"

export OVERLAY="${CLUSTER_BENCH_ROOT}/nccl_tests/overlay.img"
if [[ ! -f "${OVERLAY}" ]]; then
    echo "[pair_elem] overlay missing — run install_overlay.sh first" >&2
    exit 1
fi

pair_out="${PAIR_OUT_DIR}/${NODE_A}__${NODE_B}"
mkdir -p "${pair_out}"
COLLECTIVE="${COLLECTIVE:-all_reduce}"
BIN="/opt/nccl-tests-build/build/${COLLECTIVE}_perf"

srun --nodes=2 --ntasks-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=2 --gres=gpu:${GPUS_PER_NODE} \
     --nodelist="${NODE_A},${NODE_B}" \
    "${CLUSTER_BENCH_ROOT}/shared/launch.sh" \
    "${BIN}" -b 1M -e 1G -f 2 -n 20 -w 5 -g 1 \
    > "${pair_out}/${COLLECTIVE}.raw" 2>&1

echo "[pair_elem] done → ${pair_out}/${COLLECTIVE}.raw"
