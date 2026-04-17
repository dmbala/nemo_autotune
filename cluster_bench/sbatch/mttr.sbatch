#!/bin/bash
#SBATCH --job-name=cb_mttr
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Mean-time-to-recovery probe. Reuses the kill-and-resume flow from the
# basic_examples project (Module 08): launches a tiny training run, waits for
# the first checkpoint to land, kills the process group, relaunches, and
# measures how long the resume + re-load takes.
#
# Emits a single MTTR row into history.jsonl for continuous tracking.

set -euo pipefail
mkdir -p logs

CLUSTER_BENCH_ROOT="${CLUSTER_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${CLUSTER_BENCH_ROOT}/shared/slurm_common.sh"

KAR="${BASIC_EXAMPLES_ROOT}/08_fault_tolerance/kill_and_resume.sh"
if [[ ! -x "${KAR}" ]]; then
    echo "[mttr] ${KAR} not found — basic_examples layout moved or missing" >&2
    exit 1
fi

HOST="$(hostname -s)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${RESULTS_ROOT}/cluster_bench/mttr/${HOST}_${TS}"
mkdir -p "${OUT_DIR}"

SHAKES_BIN="${DATA_ROOT}/shakespeare/train.bin"
if [[ ! -f "${SHAKES_BIN}" ]]; then
    echo "[mttr] TinyShakespeare data missing — prep via basic_examples/01_data/tiny_shakespeare.py"
    exit 1
fi

# Run the kill-and-resume flow, timing the two phases.
START_TS="$(date +%s.%N)"
"${KAR}" "${SHAKES_BIN}" "${OUT_DIR}/ckpts" 180 > "${OUT_DIR}/run.log" 2>&1 || \
    echo "[mttr] kill_and_resume exited non-zero — check ${OUT_DIR}/run.log"
END_TS="$(date +%s.%N)"

# Extract phase-1 and phase-2 timings from the log. The kill_and_resume script
# prints "[chaos] first checkpoint: ... (after Ns)" for phase 1 and "Loading
# checkpoint from ..." near phase 2's start. Wall-clock between kill and next
# training step is the MTTR signal.
python3 - "${OUT_DIR}/run.log" "${START_TS}" "${END_TS}" "${HOST}" "${RESULTS_ROOT}/cluster_bench/history.jsonl" <<'PY'
import datetime
import json
import os
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
start = float(sys.argv[2])
end = float(sys.argv[3])
host = sys.argv[4]
hist = Path(sys.argv[5])

phase1_s = None
phase2_first_step_idx = None
kill_offset_s = None

if log_path.exists():
    text = log_path.read_text()
    m = re.search(r"first checkpoint:.*\(after (\d+)s\)", text)
    if m:
        phase1_s = int(m.group(1))
    # Phase 2 starts after "PHASE 2: relaunching" marker; grab first "step N" after it.
    phase2_section = text.split("PHASE 2:", 1)[-1] if "PHASE 2:" in text else ""
    m2 = re.search(r"step (\d+) \| epoch", phase2_section)
    if m2:
        phase2_first_step_idx = int(m2.group(1))

hist.parent.mkdir(parents=True, exist_ok=True)
with hist.open("a") as f:
    f.write(json.dumps({
        "kind": "mttr",
        "hostname": host,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "total_wall_s": round(end - start, 2),
        "phase1_to_first_ckpt_s": phase1_s,
        "phase2_resume_from_step": phase2_first_step_idx,
    }) + "\n")
print(f"[mttr] wall={end-start:.1f}s phase1={phase1_s}s resume_from_step={phase2_first_step_idx}")
PY
echo "[mttr] done → ${OUT_DIR}"
