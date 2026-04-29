"""Offline: materialize a Hugging-Face-loadable directory from an Automodel
checkpoint directory (sharded DCP safetensors).

Use this when a training run was launched with ``checkpoint.save_consolidated: false``
— the run only wrote sharded shards. Module 05 inference and Module 07 eval
expect a flat HF ``save_pretrained`` directory.

If your run already set ``save_consolidated: true``, the recipe wrote a
``.../model/consolidated/`` directory automatically and you don't need this.

Usage:
    shared/launch.sh python 03_distributed/export_to_hf.py \
        --in  $CKPT_ROOT/gpt2_124m/epoch_0_step_100/model \
        --out $CKPT_ROOT/gpt2_124m/epoch_0_step_100/model/consolidated
"""
from __future__ import annotations

import argparse
from pathlib import Path

# The recipe's consolidation helper handles DCP → HF safetensors conversion,
# including stitching shards and emitting config.json / generation_config.json.
from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path,
                    help="Shard dir containing shard-*.safetensors (e.g. .../epoch_<N>_step_<S>/model)")
    ap.add_argument("--out", dest="dst", required=True, type=Path,
                    help="Output HF directory (will be created)")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"input dir does not exist: {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)

    # Single-process consolidation: one rank does everything. (For very large
    # models, launch via torchrun and the helper will shard the work.)
    consolidate_safetensors_files_on_every_rank(
        input_dir=str(args.src),
        output_dir=str(args.dst),
        rank=0,
        world_size=1,
    )
    print(f"[export_to_hf] wrote HF-loadable dir → {args.dst}")
    print(f"[export_to_hf] verify: AutoModelForCausalLM.from_pretrained('{args.dst}')")


if __name__ == "__main__":
    main()
