#!/bin/bash
# Build nccl-tests into a writable Singularity overlay for use with shared/launch.sh.
# Runs on a login node; compute nodes typically lack outbound internet.
#
# Usage:
#   bash nccl_tests/install_overlay.sh             # default 2 GB overlay
#   bash nccl_tests/install_overlay.sh 4096         # custom size (MB)
#
# Idempotent: skips the overlay creation if overlay.img already exists but always
# rebuilds the binaries (fast, ~2 min) so you can pick up any upstream change.
#
# Binaries land under /opt/nccl-tests-build/build/ inside the overlay. Build
# with MPI=0 so we can launch via torchrun / srun without Open MPI version coupling.

set -euo pipefail

CLUSTER_BENCH_ROOT="${CLUSTER_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OVERLAY="${CLUSTER_BENCH_ROOT}/nccl_tests/overlay.img"
SIZE_MB="${1:-2048}"
NCCL_TESTS_REF="${NCCL_TESTS_REF:-v2.13.13}"

if [[ ! -f "${OVERLAY}" ]]; then
    "${CLUSTER_BENCH_ROOT}/shared/user_overlay.sh" "${OVERLAY}" "${SIZE_MB}"
fi

# Build nccl-tests inside the container, writing to the overlay.
# MPI=0 because the container's Open MPI isn't guaranteed to match Slurm's,
# and torchrun / srun handle rendezvous anyway.
OVERLAY="${OVERLAY}" "${CLUSTER_BENCH_ROOT}/shared/launch.sh" bash -lc '
set -euo pipefail
SRC=/opt/nccl-tests-build
if [[ ! -d "$SRC/.git" ]]; then
    rm -rf "$SRC"
    git clone --depth=1 --branch "'"${NCCL_TESTS_REF}"'" https://github.com/NVIDIA/nccl-tests.git "$SRC" || \
        git clone --depth=1 https://github.com/NVIDIA/nccl-tests.git "$SRC"
fi
cd "$SRC"
git pull --ff-only 2>/dev/null || true
make clean 2>/dev/null || true
make -j$(nproc) MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr
ls build/*_perf
'

echo
echo "[nccl_tests] overlay ready: ${OVERLAY}"
echo "[nccl_tests] binaries available at /opt/nccl-tests-build/build/ inside the container"
echo "[nccl_tests] usage: OVERLAY=${OVERLAY} shared/launch.sh /opt/nccl-tests-build/build/all_reduce_perf -b 1M -e 1G -f 2 -g <gpus>"
