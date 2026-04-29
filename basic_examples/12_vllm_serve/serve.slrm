#!/bin/bash
#SBATCH --job-name=ws12_vllm_serve
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Starts vLLM serving a Qwen3-0.6B (or a provided path), fires a single client
# query against it, then shuts down. Adjust MODEL / PROMPT via env.

set -euo pipefail
mkdir -p logs

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${WORKSHOP_ROOT}/shared/slurm_common.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-Explain photosynthesis in one paragraph.}"
OVERLAY="${OVERLAY:-${WORKSHOP_ROOT}/12_vllm_serve/overlay.img}"

echo "[serve] model=${MODEL}"
echo "[serve] endpoint=http://$(hostname):${PORT}/v1"

# Start vLLM in the background.
OVERLAY="${OVERLAY}" "${WORKSHOP_ROOT}/12_vllm_serve/serve.sh" "${MODEL}" "${PORT}" \
    --max-model-len 2048 &
SERVER_PID=$!

# Wait for /v1/models to return 200 (vLLM can take up to ~2 min on cold start).
echo "[serve] waiting for server readiness ..."
for i in $(seq 1 120); do
    if curl -fsS "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "[serve] server ready after ${i}s"
        break
    fi
    sleep 1
done

# Smoke client query.
OVERLAY="${OVERLAY}" "${WORKSHOP_ROOT}/shared/launch.sh" python \
    "${WORKSHOP_ROOT}/12_vllm_serve/client_example.py" \
    --endpoint "http://localhost:${PORT}/v1" \
    --model "${MODEL}" \
    --prompt "${PROMPT}" \
    --max-tokens 150

echo "[serve] done, shutting down server (pid=${SERVER_PID})"
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
