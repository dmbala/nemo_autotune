#!/bin/bash
#SBATCH --job-name=cb_storage
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Rotates the same training benchmark across three storage tiers so we can
# compare DCP checkpoint save throughput end-to-end.
#
# Tiers (override via TIERS env var, space-separated):
#   /tmp                         — node-local SSD
#   /n/netscratch/...            — parallel netscratch
#   /n/holylfs06/.../scratch/    — holylfs
#
# Each tier: 1 training run, 4 save events (ckpt_every_steps=10, max_steps=40).
# Result JSONs land under $RESULTS_ROOT/cluster_bench/storage/<ts>/<tier>/.
# `analysis/scrape_metrics.py` collates save durations from the recipe's log.

set -euo pipefail
mkdir -p logs

CLUSTER_BENCH_ROOT="${CLUSTER_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${CLUSTER_BENCH_ROOT}/shared/slurm_common.sh"

HOST="$(hostname -s)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_BASE="${RESULTS_ROOT}/cluster_bench/storage/${TS}"
mkdir -p "${OUT_BASE}"

# Defaults — user-overridable. Kept as whitespace-separated list of (tag=root)
# pairs so the tag becomes the subdir name and root is the actual path prefix.
TIERS="${TIERS:-tmp=/tmp netscratch=${SCRATCH_ROOT} holylfs=/n/holylfs06/LABS/kempner_shared/Everyone/${USER}_cb_storage}"
CONFIG="${CLUSTER_BENCH_ROOT}/configs/storage.yaml"

for tier_spec in ${TIERS}; do
    tag="${tier_spec%%=*}"
    root="${tier_spec#*=}"
    ckpt_dir="${root}/cb_storage_${HOST}_${TS}"
    tier_out="${OUT_BASE}/${tag}"
    mkdir -p "${tier_out}"
    mkdir -p "${ckpt_dir}" 2>/dev/null || { echo "[storage] cannot write to ${root}; skipping"; continue; }

    echo
    echo "=== [storage] tier=${tag} root=${root} ==="
    # Time the whole run end-to-end; the recipe's internal log will show per-save timings.
    (time "${CLUSTER_BENCH_ROOT}/shared/launch.sh" automodel -c "${CONFIG}" benchmark llm \
        --checkpoint.checkpoint_dir="${ckpt_dir}" \
        --benchmark.json_output_path="${tier_out}/benchmark_results.json" \
        --nproc-per-node=1) > "${tier_out}/stdout.log" 2>&1 || echo "[storage] ${tag} run failed"

    # Record the ckpt dir size to compute throughput later
    total_bytes=$(du -sb "${ckpt_dir}" 2>/dev/null | awk '{print $1}' || echo 0)
    echo "${total_bytes}" > "${tier_out}/total_bytes.txt"

    # Clean up the checkpoint dir — we measured, we don't need the bytes
    rm -rf "${ckpt_dir}"
done

echo
echo "[storage] all tiers done → ${OUT_BASE}"
