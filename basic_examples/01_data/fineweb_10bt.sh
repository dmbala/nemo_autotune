#!/bin/bash
# Tokenize FineWeb-10BT (HF hub) into .bin shards readable by NanogptDataset.
# Wraps /opt/Automodel/tools/nanogpt_data_processor.py.
#
# Usage:
#   01_data/fineweb_10bt.sh [output_dir] [max_tokens]
#
# Defaults: writes ~500M tokens to $DATA_ROOT/fineweb_500M/ in ~20 min on 32 CPUs.
set -euo pipefail

WORKSHOP_ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/n/netscratch/kempner_dev/Lab/${USER}/Agent/nemo/runs/data}"

OUT_DIR="${1:-${DATA_ROOT}/fineweb_500M}"
MAX_TOKENS="${2:-500M}"

mkdir -p "${OUT_DIR}"

"${WORKSHOP_ROOT}/shared/launch.sh" python /opt/Automodel/tools/nanogpt_data_processor.py \
    --dataset HuggingFaceFW/fineweb \
    --set-name sample-10BT \
    --max-tokens "${MAX_TOKENS}" \
    --tokenizer gpt2 \
    --output-dir "${OUT_DIR}"

echo "[fineweb] done → ${OUT_DIR}"
ls -lh "${OUT_DIR}" | head
