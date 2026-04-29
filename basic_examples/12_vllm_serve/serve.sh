#!/bin/bash
# Start vLLM's OpenAI-compatible server.
#
# Usage:
#   12_vllm_serve/serve.sh <model> [port=8000] [--max-model-len 4096] [other vllm flags]
#
# Examples:
#   # Serve a Hub model:
#   12_vllm_serve/serve.sh Qwen/Qwen3-0.6B 8000
#
#   # Serve a locally-trained consolidated checkpoint:
#   12_vllm_serve/serve.sh $CKPT_ROOT/trackB_qwen3_0p6b_lora_squad/epoch_0_step_299/model/consolidated 8000
#
# The server binds 0.0.0.0:<port>. On Slurm, use the node's hostname (scontrol
# show hostnames $SLURM_JOB_NODELIST) — localhost only works from the same node.

set -euo pipefail

MODEL="${1:?usage: serve.sh <model_or_path> [port] [extra flags]}"
PORT="${2:-8000}"
if (( $# >= 2 )); then shift 2; else shift 1; fi

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OVERLAY="${OVERLAY:-${WORKSHOP_ROOT}/12_vllm_serve/overlay.img}"

OVERLAY="${OVERLAY}" "${WORKSHOP_ROOT}/shared/launch.sh" \
    vllm serve "${MODEL}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --dtype bfloat16 \
        --trust-remote-code \
        "$@"
