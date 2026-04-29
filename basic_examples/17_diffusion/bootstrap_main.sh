#!/bin/bash
# Bootstrap the Automodel `main` branch under `17_diffusion/_automodel_main` and
# set up the bind-mount env so TrainDiffusionRecipe is available at runtime.
#
# The 26.02 container ships only `wan2.2/wan_generate.py` under examples/diffusion
# and has no `nemo_automodel.recipes.diffusion.train` module. Upstream main added
# a full pretrain + finetune + generate stack. Until a newer SIF is available,
# we shadow /opt/Automodel with a writable clone of main.
#
# Usage:
#   bash 17_diffusion/bootstrap_main.sh            # clone or fast-forward
#
# After bootstrapping, every shared/launch.sh invocation that sets
# EXTRA_BINDS="${WORKSHOP_ROOT}/17_diffusion/_automodel_main:/opt/Automodel" will
# pick up the new recipe. The finetune/pretrain/generate sbatches below do this
# automatically.

set -euo pipefail

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CLONE_DIR="${WORKSHOP_ROOT}/17_diffusion/_automodel_main"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/NVIDIA-NeMo/Automodel.git}"

if [[ -d "${CLONE_DIR}/.git" ]]; then
    echo "[bootstrap] updating ${CLONE_DIR}"
    git -C "${CLONE_DIR}" fetch --depth=1 origin main
    git -C "${CLONE_DIR}" reset --hard origin/main
else
    # Interrupted clones leave a non-empty dir with no .git; git refuses to
    # clone into it. Remove the stub so the retry succeeds.
    if [[ -e "${CLONE_DIR}" ]]; then
        echo "[bootstrap] ${CLONE_DIR} exists but is missing .git — removing stub"
        rm -rf "${CLONE_DIR}"
    fi
    echo "[bootstrap] cloning main into ${CLONE_DIR}"
    git clone --depth=1 --branch main "${UPSTREAM_URL}" "${CLONE_DIR}"
fi

echo "[bootstrap] done."
echo "[bootstrap] To run the diffusion recipe, have EXTRA_BINDS set so the shadow mount is active:"
echo "  export EXTRA_BINDS=${CLONE_DIR}:/opt/Automodel"
echo "  shared/launch.sh python /opt/Automodel/examples/diffusion/finetune/finetune.py --config <yaml>"
