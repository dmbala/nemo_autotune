"""Pre-warm the HF cache on a login node so compute nodes don't need Hub access.

Useful if your Slurm compute nodes have restricted internet (common). Run this
once from a login node with HF_HOME pointed at a shared path that compute
nodes can read.

Usage:
    shared/launch.sh python 04_finetune/download_hf_model.py \
        --model Qwen/Qwen3-0.6B

    # gated model (set HF_TOKEN first):
    HF_TOKEN=hf_... shared/launch.sh python 04_finetune/download_hf_model.py \
        --model meta-llama/Llama-3.2-1B
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id, e.g. Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("HF_HOME")
    print(f"[download] HF_HOME={cache_dir}")
    print(f"[download] pulling {args.model} ...")

    path = snapshot_download(
        repo_id=args.model,
        token=os.environ.get("HF_TOKEN"),
        local_files_only=False,
    )
    print(f"[download] cached at: {path}")
    print("[download] compute nodes can now load this model without internet.")


if __name__ == "__main__":
    main()
