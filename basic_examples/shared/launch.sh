#!/bin/bash
# Entry point for running any command inside the NeMo-Automodel Singularity container.
#
# Usage:
#   shared/launch.sh <any command + args>
#
# Example:
#   shared/launch.sh automodel -c 02_pretrain/configs/tiny_gpt2_shakespeare.yaml pretrain LLM
#   shared/launch.sh python 00_setup/smoke_test.py
#
# Env overrides:
#   SIF            path to the .sif (default: the shared fixed image)
#   HF_HOME        HF hub cache root. If unset, slurm_common.sh picks it; if
#                  running launch.sh directly without slurm_common, we fall back
#                  to $SCRATCH_ROOT/.hf where SCRATCH_ROOT itself defaults to
#                  /n/netscratch/kempner_dev/Lab/$USER/Agent/nemo/runs.
#   HF_TOKEN       set for gated models (e.g. Llama)
#   EXTRA_BINDS    comma-separated extra --bind specs passed to singularity
#                  (each spec is either `host_path` or `host_path:container_path`)
#   OVERLAY        path to a writable overlay image (used by Module 07 for lm-eval install)

set -euo pipefail

: "${SIF:=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/nemo/nemo-automodel-26.02-fixed.sif}"
: "${SCRATCH_ROOT:=/n/netscratch/kempner_dev/Lab/${USER}/Agent/nemo/runs}"
: "${HF_HOME:=${SCRATCH_ROOT}/.hf}"
: "${EXTRA_BINDS:=}"
: "${OVERLAY:=}"

mkdir -p "${HF_HOME}"

BINDS="/n/netscratch,/n/holylfs06"
if [[ -n "${EXTRA_BINDS}" ]]; then
    BINDS="${BINDS},${EXTRA_BINDS}"
fi

OVERLAY_ARG=()
if [[ -n "${OVERLAY}" ]]; then
    OVERLAY_ARG=(--overlay "${OVERLAY}")
fi

# Override the host's CA-bundle env vars — the host points at
# /etc/ssl/certs/ca-bundle.crt (RHEL-style) which doesn't exist inside the
# Ubuntu-based NGC container. Use the container's bundle instead.
ENV_FWD="HF_HOME=${HF_HOME}"
ENV_FWD="${ENV_FWD},SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
ENV_FWD="${ENV_FWD},SSL_CERT_DIR=/etc/ssl/certs"
ENV_FWD="${ENV_FWD},CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"
ENV_FWD="${ENV_FWD},REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"
if [[ -n "${HF_TOKEN:-}" ]]; then
    ENV_FWD="${ENV_FWD},HF_TOKEN=${HF_TOKEN}"
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
    ENV_FWD="${ENV_FWD},PYTHONPATH=${PYTHONPATH}"
fi

exec singularity exec --nv \
    --bind "${BINDS}" \
    "${OVERLAY_ARG[@]}" \
    --env "${ENV_FWD}" \
    "${SIF}" \
    bash -lc "source /opt/venv/env.sh && exec \"\$@\"" _ "$@"
