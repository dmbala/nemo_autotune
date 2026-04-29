#!/bin/bash
# All-pairs 2-node allreduce driver. Instead of submitting N*(N-1)/2 separate
# jobs (which crushes the scheduler at cluster scale), we write the pair list
# to a file and submit a single sbatch array where each element handles one
# pair. Scales cleanly to hundreds of nodes.
#
# Usage:
#   cluster_bench/nccl_tests/pair_matrix.sh "node1 node2 node3 node4"
#
# Each array element runs the 2-node all_reduce benchmark on its assigned pair.
# Collate afterwards with analysis/scrape_metrics.py.

set -euo pipefail

NODES="${1:?usage: pair_matrix.sh \"node1 node2 node3 ...\"}"
CLUSTER_BENCH_ROOT="${CLUSTER_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

read -ra NODE_ARR <<< "${NODES}"
N=${#NODE_ARR[@]}
if (( N < 2 )); then
    echo "[pair_matrix] need at least 2 nodes, got ${N}" >&2
    exit 1
fi

# Sanity-check each hostname looks like a valid Slurm identifier (avoids
# shell-injection or Slurm parse errors from stray whitespace / globs).
for h in "${NODE_ARR[@]}"; do
    if ! [[ "${h}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[pair_matrix] invalid hostname: '${h}'" >&2
        exit 2
    fi
done

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/kempner_dev/Lab/${USER}/Agent/nemo/runs/results}"
OUT_DIR="${RESULTS_ROOT}/cluster_bench/nccl_pair_matrix/${TS}"
mkdir -p "${OUT_DIR}"

# Materialize the pair list once. Array element k → line k+1 in this file.
PAIRS="${OUT_DIR}/pairs.txt"
: > "${PAIRS}"
for (( i=0; i<N; i++ )); do
    for (( j=i+1; j<N; j++ )); do
        printf '%s %s\n' "${NODE_ARR[i]}" "${NODE_ARR[j]}" >> "${PAIRS}"
    done
done
n_pairs=$(wc -l < "${PAIRS}")
max_idx=$((n_pairs - 1))

echo "[pair_matrix] ${N} nodes, ${n_pairs} pairs → ${OUT_DIR}"
echo "[pair_matrix] submitting one sbatch array job"

sbatch --array=0-${max_idx} \
    --output="${OUT_DIR}/pair_%a.out" \
    --error="${OUT_DIR}/pair_%a.err" \
    --job-name="cb_pair_matrix" \
    --export=ALL,PAIRS_FILE="${PAIRS}",PAIR_OUT_DIR="${OUT_DIR}" \
    "${CLUSTER_BENCH_ROOT}/nccl_tests/_pair_elem.slrm"

echo
echo "[pair_matrix] array job submitted. When it completes, collate with:"
echo "  shared/launch.sh python ${CLUSTER_BENCH_ROOT}/analysis/scrape_metrics.py \\"
echo "      --results ${RESULTS_ROOT}/cluster_bench --out results/history.jsonl"
