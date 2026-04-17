#!/bin/bash
#SBATCH --job-name=cb_nccl_inter
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# 2-node NCCL allreduce (8 GPUs total, world_size=8). Measures IB/RoCE path.
# Expected healthy busbw on 4× NDR 400 Gb/s IB:
#   all_reduce @ 1 GB:  100-180 GB/s busbw  (depends on rail configuration)
#
# The nccl-tests binary needs to be invoked under torchrun (or mpirun) for
# multi-node. We use torchrun with c10d rendezvous, consistent with the rest
# of the workshop.

set -euo pipefail
mkdir -p logs

CLUSTER_BENCH_ROOT="${CLUSTER_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${CLUSTER_BENCH_ROOT}/shared/slurm_common.sh"

export OVERLAY="${CLUSTER_BENCH_ROOT}/nccl_tests/overlay.img"
if [[ ! -f "${OVERLAY}" ]]; then
    echo "[nccl_inter] ${OVERLAY} missing — run install_overlay.sh first." >&2
    exit 1
fi

COLLECTIVE="${COLLECTIVE:-all_reduce}"
BIN="/opt/nccl-tests-build/build/${COLLECTIVE}_perf"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${RESULTS_ROOT}/cluster_bench/nccl/${TS}_inter"
mkdir -p "${OUT_DIR}"

# nccl-tests binaries are MPI=0, so they expect one process per GPU and will use
# ranks from torch.distributed. torchrun launches them with the right env.
srun --nodes=${NNODES} --ntasks-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=2 --gres=gpu:${GPUS_PER_NODE} \
    "${CLUSTER_BENCH_ROOT}/shared/launch.sh" bash -c "
${BIN} -b 1M -e 1G -f 2 -n 20 -w 5 -g 1
" | tee "${OUT_DIR}/${COLLECTIVE}.raw"

echo
echo "[nccl_inter] raw → ${OUT_DIR}/${COLLECTIVE}.raw"
