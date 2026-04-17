#!/bin/bash
#SBATCH --job-name=cb_nccl_intra
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Run all 6 NCCL collectives on a single node. Measures NVLink / NVSwitch path.
# Expected healthy busbw on 4× H100 NVLink:
#   all_reduce:       ~300 GB/s at 1 GB
#   all_gather:       ~300 GB/s at 1 GB
#   reduce_scatter:   ~300 GB/s at 1 GB
#   alltoall:         ~200 GB/s at 1 GB
#
# Override NCCL_TESTS via env: NCCL_TESTS="all_reduce all_gather" sbatch ...

set -euo pipefail
mkdir -p logs

CLUSTER_BENCH_ROOT="${CLUSTER_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${CLUSTER_BENCH_ROOT}/shared/slurm_common.sh"

NCCL_TESTS="${NCCL_TESTS:-all_reduce all_gather reduce_scatter alltoall broadcast sendrecv}"

for c in ${NCCL_TESTS}; do
    echo
    echo "=== [nccl_intra] ${c} ==="
    "${CLUSTER_BENCH_ROOT}/nccl_tests/run_collective.sh" "${c}" -g "${GPUS_PER_NODE}"
done
