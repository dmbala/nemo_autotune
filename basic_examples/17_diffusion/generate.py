"""Distributed diffusion inference via ``NeMoAutoDiffusionPipeline``.

Loads any ``diffusers.DiffusionPipeline`` and parallelizes the heavy transformer
components across the torchrun world via Automodel's ``FSDP2Manager``. Default
target is Wan2.2-T2V-A14B (14B-param video DiT) which fits 4×H100 in bf16 with
TP=4 on the transformer stages.

Shape of the call, generalized from
``/opt/Automodel/examples/diffusion/wan2.2/wan_generate.py``:

    torchrun --nproc-per-node=4 17_diffusion/generate.py \\
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \\
        --prompt "..." --tp-size 4

The script must be launched via torchrun (or srun torchrun), not via the
``automodel`` CLI — the recipe path only handles LLM/VLM finetune/pretrain/kd.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import torch
import torch.distributed as dist
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

# Upstream's wrapper lives under a leading-underscore module (private API surface
# today, may be promoted later — pin here and track upstream).
from nemo_automodel._diffusers import NeMoAutoDiffusionPipeline
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging


_DEFAULT_PROMPT = (
    "A cinematic aerial shot of a snow-capped mountain range at sunrise, clouds "
    "drifting through the valleys, the camera slowly descending toward a winding "
    "river. Soft warm light."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Distributed diffusion generation")
    ap.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                    help="HF repo id of a diffusers pipeline.")
    ap.add_argument("--prompt", default=_DEFAULT_PROMPT)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--num-inference-steps", type=int, default=20,
                    help="Denoising / flow-matching steps. Fewer = faster but lower quality.")
    ap.add_argument("--guidance-scale", type=float, default=4.0)
    ap.add_argument("--guidance-scale-2", type=float, default=3.0,
                    help="Wan2.2 second-stage CFG. Ignored by pipelines that don't use it.")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--output", default="output.mp4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--cp-size", type=int, default=1)
    ap.add_argument("--pp-size", type=int, default=1)
    ap.add_argument("--dp-size", type=int, default=1)
    return ap.parse_args()


def _dit_component_names(model_id: str) -> tuple[str, ...]:
    """Return which of {'transformer', 'transformer_2'} the pipeline actually has.

    Image pipelines (Flux, SD3, SDXL) only expose 'transformer'; Wan-family video
    pipelines expose both. Checking the model_index before building
    ``parallel_scheme`` avoids a post-download failure when we reference a
    component the pipeline doesn't ship.
    """
    try:
        path = snapshot_download(model_id, allow_patterns=["model_index.json"])
        with open(os.path.join(path, "model_index.json")) as f:
            index = json.load(f)
    except Exception as exc:
        logging.warning("could not read model_index.json for %s (%s); assuming transformer only",
                        model_id, exc)
        return ("transformer",)
    return tuple(n for n in ("transformer", "transformer_2") if n in index)


def main() -> None:
    args = parse_args()
    # Bind this rank's CUDA device *before* NCCL init so the backend uses it.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    initialize_distributed(backend="nccl", timeout_minutes=15)
    setup_logging()

    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    dp_rank = local_rank // (args.tp_size * args.cp_size * args.pp_size)

    fsdp2_manager = FSDP2Manager(
        dp_size=args.dp_size, tp_size=args.tp_size,
        cp_size=args.cp_size, pp_size=args.pp_size,
        backend="nccl", world_size=world_size, use_hf_tp_plan=False,
    )

    parallel_scheme = {name: fsdp2_manager for name in _dit_component_names(args.model)}
    logging.info("[setup] will parallelize components=%s", tuple(parallel_scheme))

    pipe = NeMoAutoDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device=device,
        parallel_scheme=parallel_scheme,
    )
    logging.info("[setup] pipeline sharded tp=%d cp=%d pp=%d dp=%d",
                 args.tp_size, args.cp_size, args.pp_size, args.dp_size)
    dist.barrier()

    seed = args.seed + dp_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = pipe(
            prompt=args.prompt,
            height=args.height, width=args.width, num_frames=args.num_frames,
            guidance_scale=args.guidance_scale, guidance_scale_2=args.guidance_scale_2,
            num_inference_steps=args.num_inference_steps,
        ).frames[0]

    if dist.get_rank() == 0:
        export_to_video(out, args.output, fps=args.fps)
        logging.info("[done] wrote %s (%dx%d, %d frames, %d steps)",
                     args.output, args.width, args.height, args.num_frames, args.num_inference_steps)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
