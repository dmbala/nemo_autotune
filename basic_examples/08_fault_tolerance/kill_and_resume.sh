#!/bin/bash
# Kill-and-resume demo. Two phases against the *same* checkpoint_dir:
#
#  Phase 1: launch training, kill it with SIGKILL after N seconds.
#  Phase 2: launch again with the same config. restore_from: LATEST picks up
#           from the last saved step.
#
# Expected outcome: phase-2 log should show "Loading checkpoint from
# .../epoch_0_step_K" and the step counter resumes at K+1 instead of 0.
#
# Usage:
#   08_fault_tolerance/kill_and_resume.sh <train.bin_glob> <ckpt_dir> [max_wait_seconds=180]
#
# Phase 1 waits for the first checkpoint to land (proving training actually
# started), then kills the runner. That way the demo works regardless of
# container spin-up / hardware speed, provided the checkpoint interval is
# short enough to land within max_wait_seconds.

set -euo pipefail

TRAIN_BIN="${1:?usage: kill_and_resume.sh <train.bin_glob> <ckpt_dir> [max_wait]}"
CKPT_DIR="${2:?ckpt_dir}"
# Seconds to wait for the first checkpoint before giving up. Container
# spin-up + python imports eat ~15 s before training even starts, so be
# generous. The script actually kills as soon as the first ckpt lands +
# a few steps, not at this deadline.
MAX_WAIT="${3:-180}"

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${WORKSHOP_ROOT}/08_fault_tolerance/configs/tiny_resume.yaml"
LAUNCH="${WORKSHOP_ROOT}/shared/launch.sh"

rm -rf "${CKPT_DIR}"
mkdir -p "${CKPT_DIR}"

echo
echo "============================================================"
echo "  PHASE 1: launching training, SIGKILL after first checkpoint"
echo "============================================================"
echo

# Launch phase 1 in its own session so the entire process group (shell →
# singularity → torchrun → python) can be killed as one unit.
setsid "${LAUNCH}" automodel -c "${CONFIG}" pretrain llm \
    --dataset.file_pattern="${TRAIN_BIN}" \
    --checkpoint.checkpoint_dir="${CKPT_DIR}" \
    --nproc-per-node=1 &
RUNNER_PID=$!
# Small settle so `ps` sees the post-setsid PGID, not the parent shell's.
sleep 1
RUNNER_PGID=$(ps -o pgid= -p "${RUNNER_PID}" | tr -d ' ')
if [[ -z "${RUNNER_PGID}" || "${RUNNER_PGID}" == "$$" ]]; then
    echo "[chaos] failed to capture child PGID (got '${RUNNER_PGID}', shell pid $$) — aborting rather than risk killing ourselves" >&2
    kill -KILL "${RUNNER_PID}" 2>/dev/null || true
    exit 1
fi
echo "[chaos] runner PID=${RUNNER_PID} PGID=${RUNNER_PGID}"

# Wait for the first checkpoint to land (proof training has started).
echo "[chaos] waiting up to ${MAX_WAIT}s for first checkpoint ..."
waited=0
while [[ ! -L "${CKPT_DIR}/LATEST" ]]; do
    sleep 2
    waited=$((waited + 2))
    if ! kill -0 "${RUNNER_PID}" 2>/dev/null; then
        echo "[chaos] runner exited before saving any checkpoint (elapsed ${waited}s)" >&2
        exit 1
    fi
    if (( waited >= MAX_WAIT )); then
        echo "[chaos] timed out after ${MAX_WAIT}s waiting for first checkpoint" >&2
        kill -KILL -"${RUNNER_PGID}" 2>/dev/null || true
        exit 1
    fi
done
FIRST_CKPT=$(readlink "${CKPT_DIR}/LATEST")
echo "[chaos] first checkpoint: ${FIRST_CKPT} (after ${waited}s)"

# Let training run a few more steps past the checkpoint, then kill.
sleep 3
echo "[chaos] ckpts at time of kill:"
ls -d "${CKPT_DIR}"/epoch_*_step_* 2>/dev/null | sort
echo "[chaos] sending SIGKILL to process group ${RUNNER_PGID}"
kill -KILL -"${RUNNER_PGID}" 2>/dev/null || true
wait "${RUNNER_PID}" 2>/dev/null || true
sleep 2

echo
echo "============================================================"
echo "  PHASE 2: relaunching — expecting restore_from: LATEST"
echo "============================================================"
echo

LATEST_TARGET=$(readlink "${CKPT_DIR}/LATEST" 2>/dev/null || echo "<none>")
echo "[resume] LATEST -> ${LATEST_TARGET}"

"${LAUNCH}" automodel -c "${CONFIG}" pretrain llm \
    --dataset.file_pattern="${TRAIN_BIN}" \
    --checkpoint.checkpoint_dir="${CKPT_DIR}" \
    --nproc-per-node=1
