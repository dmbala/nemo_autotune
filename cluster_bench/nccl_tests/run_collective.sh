#!/bin/bash
# Run one nccl-tests collective and append a parsed JSON row to history.jsonl.
#
# Usage:
#   cluster_bench/nccl_tests/run_collective.sh <collective> [options]
#
# <collective> ∈ {all_reduce, all_gather, reduce_scatter, alltoall, broadcast, sendrecv}
#
# Options (passed through to the nccl-tests binary):
#   -g <num_gpus>    gpus/proc (default: all visible)
#   -b <min_size>    min message size (default: 1M)
#   -e <max_size>    max message size (default: 1G)
#   -f <step>        size-step factor (default: 2)
#   -n <iters>       iterations per size (default: 20)
#   -w <warmup>      warmup iterations (default: 5)
#
# Outputs:
#   - stdout: raw nccl-tests table
#   - $RESULTS_ROOT/cluster_bench/nccl/<ts>/<collective>.csv  (parsed)
#   - Appends one JSON row to $RESULTS_ROOT/cluster_bench/history.jsonl per size

set -euo pipefail

COLLECTIVE="${1:?usage: run_collective.sh <all_reduce|all_gather|reduce_scatter|alltoall|broadcast|sendrecv> [opts...]}"
shift

CLUSTER_BENCH_ROOT="${CLUSTER_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OVERLAY="${OVERLAY:-${CLUSTER_BENCH_ROOT}/nccl_tests/overlay.img}"
if [[ ! -f "${OVERLAY}" ]]; then
    echo "[nccl] ${OVERLAY} missing — run nccl_tests/install_overlay.sh first." >&2
    exit 1
fi

BIN="/opt/nccl-tests-build/build/${COLLECTIVE}_perf"
RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/kempner_dev/Lab/${USER}/Agent/nemo/runs/results}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname -s)"
OUT_DIR="${RESULTS_ROOT}/cluster_bench/nccl/${TS}"
mkdir -p "${OUT_DIR}"
RAW="${OUT_DIR}/${COLLECTIVE}_${HOST}.raw"
CSV="${OUT_DIR}/${COLLECTIVE}_${HOST}.csv"

# Defaults match what's useful for a cluster diagnostic: 1M to 1G, factor-2 step.
ARGS=("-b" "1M" "-e" "1G" "-f" "2" "-n" "20" "-w" "5")
# If caller didn't override -g, use all GPUs visible to the torchrun/srun rank.
GFLAG=""
if [[ " $* " != *" -g "* ]]; then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        GFLAG="-g ${SLURM_GPUS_ON_NODE}"
    fi
fi

echo "[nccl] ${COLLECTIVE}_perf on ${HOST}"
OVERLAY="${OVERLAY}" "${CLUSTER_BENCH_ROOT}/shared/launch.sh" \
    "${BIN}" "${ARGS[@]}" ${GFLAG} "$@" | tee "${RAW}"

# Parse the nccl-tests output table into CSV. The format is stable enough to
# do with sed/awk: skip header rows, keep lines with 13 whitespace-separated fields.
python3 - "${RAW}" "${CSV}" "${COLLECTIVE}" "${HOST}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_text()
csv_path = Path(sys.argv[2])
collective = sys.argv[3]
hostname = sys.argv[4]

# nccl-tests data rows look like:
#   size count type redop root oop-time oop-busbw oop-#wrong ip-time ip-busbw ip-#wrong
# (columns vary per collective; we just record size + out-of-place time + busbw).
rows = []
for line in raw.splitlines():
    parts = line.split()
    if len(parts) < 8:
        continue
    if not parts[0].isdigit():
        continue
    try:
        size = int(parts[0])
        time_us = float(parts[5])
        algbw = float(parts[6])
        busbw = float(parts[7])
    except (ValueError, IndexError):
        continue
    rows.append({"size_bytes": size, "time_us": time_us, "algbw_gbps": algbw, "busbw_gbps": busbw})

with csv_path.open("w") as f:
    f.write("size_bytes,time_us,algbw_gbps,busbw_gbps\n")
    for r in rows:
        f.write(f"{r['size_bytes']},{r['time_us']},{r['algbw_gbps']},{r['busbw_gbps']}\n")

history = os.environ.get("HISTORY_JSONL", f"{os.environ.get('RESULTS_ROOT', '.')}/cluster_bench/history.jsonl")
Path(history).parent.mkdir(parents=True, exist_ok=True)
import datetime
with open(history, "a") as f:
    for r in rows:
        f.write(json.dumps({
            "kind": "nccl_perf",
            "hostname": hostname,
            "collective": collective,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            **r,
        }) + "\n")
print(f"[nccl] parsed {len(rows)} size points → {csv_path}")
print(f"[nccl] history appended → {history}")
PY
