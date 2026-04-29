#!/bin/bash
# Build a writable Singularity overlay and pip-install vLLM into it.
# Run once on a login node (compute nodes have restricted internet).
#
# Usage:
#   12_vllm_serve/install_overlay.sh [overlay_size_mb=8192]
#
# vLLM pulls a lot — the overlay defaults to 8 GB. Bump if you hit ENOSPC.

set -euo pipefail

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OVERLAY="${WORKSHOP_ROOT}/12_vllm_serve/overlay.img"
SIZE_MB="${1:-8192}"

if [[ ! -f "${OVERLAY}" ]]; then
    "${WORKSHOP_ROOT}/shared/user_overlay.sh" "${OVERLAY}" "${SIZE_MB}"
fi

# vLLM pins to a specific torch version, which will conflict with the
# container's torch 2.10.0a0+nvidia. --no-deps avoids the downgrade; we then
# manually pip-install the runtime deps vLLM actually needs at import time.
OVERLAY="${OVERLAY}" "${WORKSHOP_ROOT}/shared/launch.sh" bash -lc '
set -euo pipefail
pip install --no-cache-dir --no-deps vllm==0.11.* || {
    echo "[vllm] install with --no-deps failed; retrying with full dep resolution"
    pip install --no-cache-dir vllm==0.11.*
}
# Runtime deps vLLM expects but the container may not have:
pip install --no-cache-dir \
    "fastapi" "uvicorn" "openai>=1.50" "prometheus_client" \
    "ray[default]" "xformers" "tiktoken" "sentencepiece" "msgspec" \
    "py-cpuinfo" "partial-json-parser" "lark" "outlines" "opencv-python-headless" || true
# Verify vLLM imports (which exercises torch + cuda bindings):
python -c "import vllm; print(\"vllm:\", vllm.__version__)"
'

echo "[vllm] overlay ready: ${OVERLAY}"
echo "[vllm] launch with  OVERLAY=${OVERLAY} shared/launch.sh ..."
