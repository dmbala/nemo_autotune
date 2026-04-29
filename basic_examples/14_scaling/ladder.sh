#!/bin/bash
# Run a scaling-ladder step using Automodel's benchmark recipe.
# Each step exercises the parallelism knob that becomes necessary at that model size.
#
# Usage:
#   14_scaling/ladder.sh <step> [--nproc-per-node=N] [extra overrides]
#
# Examples:
#   14_scaling/ladder.sh step2 --nproc-per-node=4
#   14_scaling/ladder.sh step5 --nproc-per-node=8
#
# Output: per-run JSON summary at $RESULTS_ROOT/scaling/<step>/benchmark_results.json
# with avg_mfu_percent, tflops_per_gpu, and avg_iter_time_seconds.

set -euo pipefail

STEP="${1:?usage: ladder.sh <stepN> [extra overrides]}"
shift

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
shopt -s nullglob
matches=("${WORKSHOP_ROOT}/14_scaling/configs/${STEP}_"*.yaml)
shopt -u nullglob
if (( ${#matches[@]} == 0 )); then
    echo "No config matching ${STEP}_*.yaml under 14_scaling/configs/" >&2
    echo "Available:" >&2
    ls "${WORKSHOP_ROOT}/14_scaling/configs/" | sed 's/^/  /' >&2
    exit 1
fi
CFG="${matches[0]}"

RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/kempner_dev/Lab/${USER}/Agent/nemo/runs/results}"
OUT_DIR="${RESULTS_ROOT}/scaling/${STEP}"
mkdir -p "${OUT_DIR}"

echo "[ladder] step=${STEP}  config=$(basename ${CFG})"
echo "[ladder] results → ${OUT_DIR}/benchmark_results.json"

"${WORKSHOP_ROOT}/shared/launch.sh" automodel -c "${CFG}" benchmark llm \
    --benchmark.json_output_path="${OUT_DIR}/benchmark_results.json" \
    "$@"
