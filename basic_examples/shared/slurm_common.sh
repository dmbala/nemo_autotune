# Source this from SBATCH scripts AFTER the #SBATCH directives.
# Provides shared env setup for the workshop runs on Kempner H100/H200 nodes.

set -euo pipefail

# --- paths ----
WORKSHOP_ROOT="${WORKSHOP_ROOT:-/n/netscratch/kempner_dev/Lab/${USER}/Agent/nemo/workshop}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/n/netscratch/kempner_dev/Lab/${USER}/Agent/nemo/runs}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH_ROOT}/data}"
CKPT_ROOT="${CKPT_ROOT:-${SCRATCH_ROOT}/checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH_ROOT}/results}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/.hf}"

mkdir -p "${SCRATCH_ROOT}" "${DATA_ROOT}" "${CKPT_ROOT}" "${RESULTS_ROOT}" "${HF_HOME}"

# --- container ----
export SIF="${SIF:-/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/nemo/nemo-automodel-26.02-fixed.sif}"
LAUNCH="${WORKSHOP_ROOT}/shared/launch.sh"
[[ -x "${LAUNCH}" ]] || chmod +x "${LAUNCH}"

# --- distributed rendezvous (populated on multi-task jobs) ----
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE="${SLURM_NTASKS:-1}"
export NNODES="${SLURM_JOB_NUM_NODES:-1}"
export GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"

# --- NCCL / cluster hygiene ----
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export GODEBUG=http2client=0

echo "[slurm_common] node=$(hostname) MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${WORLD_SIZE} NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "[slurm_common] SIF=${SIF}"
echo "[slurm_common] SCRATCH_ROOT=${SCRATCH_ROOT}"
