# Module 17 — Diffusion & flow-matching models

Upstream Automodel (main branch) ships a complete **diffusion / flow-matching training stack**, covering:

| Phase | Upstream entry | Module file | Recipe |
|---|---|---|---|
| Pretrain | `examples/diffusion/pretrain/pretrain.py` | `17_diffusion/pretrain.py` | `nemo_automodel.recipes.diffusion.train.TrainDiffusionRecipe` |
| Finetune (incl. LoRA) | `examples/diffusion/finetune/finetune.py` | `17_diffusion/finetune.py` | same recipe, `mode: finetune` |
| Generate | `examples/diffusion/generate/generate.py` + `wan2.2/wan_generate.py` | `17_diffusion/generate.py` | `NeMoAutoDiffusionPipeline` + `FSDP2Manager` |

Supported model families (from upstream configs): **Flux-T2I**, **Wan2.1-T2V**, **Hunyuan-T2V**, **Wan2.2-T2V**. The training recipe uses **flow matching** (rectified-flow), not classical DDPM.

## ⚠️ Container version gap

The 26.02 SIF `/n/holylfs06/.../nemo-automodel-26.02-fixed.sif` is older than current `main`. It has:
- ✅ `NeMoAutoDiffusionPipeline` + `FSDP2Manager` (generation works)
- ✅ `diffusers 0.35.2`
- ❌ **No `nemo_automodel.recipes.diffusion.train` module** (pretrain/finetune break)

Workshop solution: `bootstrap_main.sh` clones `main` locally and the pretrain/finetune wrappers shadow-mount it over `/opt/Automodel` via Singularity `--bind`. When NVIDIA publishes a newer SIF tag with the diffusion recipe baked in, drop the bind-mount. The container code in `shared/launch.sh` already supports `EXTRA_BINDS` for exactly this case.

## One-time setup

```bash
bash 17_diffusion/bootstrap_main.sh         # git clone --depth=1 of main → 17_diffusion/_automodel_main
```

## Generation (works on the 26.02 container as-is — no bootstrap needed)

```bash
# Wan2.2-T2V-A14B, 4×H100 TP=4, rectified-flow sampler, ~5-10 min per 3-second clip:
sbatch 17_diffusion/generate.slrm

PROMPT="Neon cyberpunk street market at night." NUM_STEPS=12 sbatch 17_diffusion/generate.slrm
MODEL=Wan-AI/Wan2.1-T2V-1.3B-Diffusers NUM_STEPS=25 sbatch 17_diffusion/generate.slrm
```

`generate.py` loads any `diffusers.DiffusionPipeline`, shards the DiT components (`transformer`, `transformer_2` for Wan2.2) across the torchrun world, and produces a video.

## Finetune — LoRA on Wan2.1-T2V (needs bootstrap)

```bash
# Default: LoRA on Wan2.1-T2V with flow matching, 4 GPUs:
sbatch 17_diffusion/finetune.slrm

# Flux-T2I LoRA (image, not video):
CONFIG=finetune_flux_lora.yaml sbatch 17_diffusion/finetune.slrm

# Multinode variant (set --nodes):
CONFIG=finetune_wan2_1_t2v_multinode.yaml sbatch --nodes=2 17_diffusion/finetune.slrm
```

LoRA adapters land under `checkpoint.checkpoint_dir/<step>/adapter_model.safetensors`.

## Pretrain from scratch

```bash
# Flux-T2I pretraining — defaults assume 8 GPUs on one node:
CONFIG=pretrain_flux_flow.yaml sbatch 17_diffusion/pretrain.slrm

# Wan2.1 T2V from-scratch pretraining (default):
sbatch 17_diffusion/pretrain.slrm
```

## The flow-matching YAML surface

Diffusion configs have a different shape than LLM configs. Key blocks, from `configs/finetune_flux_lora.yaml`:

```yaml
model:
  pretrained_model_name_or_path: "black-forest-labs/FLUX.1-dev"
  model_type: "flux"                  # selects model-specific adapters
  mode: "finetune"                    # vs "pretrain"
  attention_backend: "flash"

flow_matching:                        # ← the central block
  adapter_type: "flux"
  adapter_kwargs:
    guidance_scale: 1
    use_guidance_embeds: false
  timestep_sampling: "logit_normal"   # "uniform" for pretrain, "logit_normal" for finetune
  logit_mean: 0.0
  logit_std: 1.0
  flow_shift: 1.0
  num_train_timesteps: 1000
  use_loss_weighting: false
  i2v_prob: 0.0                       # image→video mix for T2V pretraining

fsdp:                                 # ← note: `fsdp` here, not `distributed`
  dp_size: 8
  tp_size: 1
  cp_size: 1
  pp_size: 1
  activation_checkpointing: false
  cpu_offload: false

optim:                                # ← `optim` not `optimizer`
  learning_rate: 2e-5
  optimizer:
    weight_decay: 0.01
    betas: [0.9, 0.999]

peft:                                 # same PeftConfig class as LLM modules
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  dim: 64
  alpha: 64
  target_modules: ["*.attn.to_q", "*.attn.to_k", ...]

data:
  dataloader:
    _target_: nemo_automodel.components.datasets.diffusion.build_text_to_image_multiresolution_dataloader
    cache_dir: PATH_TO_YOUR_DATA
    base_resolution: [512, 512]
```

Two things to fill in before launching any training config:
1. **`data.dataloader.cache_dir`** → path to your preprocessed video/image data.
2. **`checkpoint.checkpoint_dir`** → where to write ckpts.

The recipe's data pipeline is different from the LLM side — it expects pre-tokenized latents + text embeddings, typically produced by a pre-processing pass over your image/video dataset. See `docs/diffusion/` in the upstream repo (after `bootstrap_main.sh`) for data-prep tooling.

## Flow matching vs diffusion — what's the difference?

- **Diffusion (DDPM/DDIM)**: learn a noise predictor ε_θ(x_t, t); invert the forward noising step-by-step.
- **Flow matching / rectified flow**: learn a vector field v_θ(x, t) that maps noise → data along straight paths. Inference is Euler integration. Typically 4–12 sampling steps at parity quality vs 20–50 for DDIM.

Flux, Wan2.1/2.2, SD3 all use flow-matching schedulers. That's what the `flow_matching:` block configures. See [`schedulers.md`](./schedulers.md) for the sampler cheat sheet and step-count guide.

## Gotchas

- **Diffusion training needs real data.** Unlike LLM benchmark configs that can use `MockIterableDataset`, the diffusion recipe expects `cache_dir` filled with pre-processed latents. Workshop configs leave `cache_dir: PATH_TO_YOUR_DATA` as a sentinel — replace with your path before launch.
- **`fsdp.dp_size`** — single-node configs now set `dp_size: none` (auto-resolve to world_size). The multi-node variant hardcodes `dp_size: 16` for 2×8 = 16 ranks; adjust if you run on different geometry.
- **Flash attention requires `attention_backend: flash` + torch built with flash-attn.** The container has it. SDPA fallback roughly doubles activation memory.
- **`PATH_TO_YOUR_CKPT_DIR`** sentinel is overridden by `finetune.sh` / `pretrain.sh` via `--checkpoint.checkpoint_dir=<ckpt_dir>` (2nd positional arg). The sbatches default to `$CKPT_ROOT/diffusion_<config-basename>/`.
- **Bootstrap clone is ~200 MB** (shallow). 10-30 s the first time. Re-run to fast-forward when upstream advances.
- **The shadow bind-mount shadows the entire `/opt/Automodel`**, including LLM recipes. Usually fine (main is a superset) but unset `EXTRA_BINDS` to revert to the container's baked-in copy.

## What's smoke-verified

- ✅ `diffusers 0.35.2`, `NeMoAutoDiffusionPipeline`, `FSDP2Manager`, `AutoencoderKLWan` importable.
- ✅ Upstream `pretrain.py`/`finetune.py` copied verbatim — 29-line launchers.
- ⏸ `TrainDiffusionRecipe` itself not in the 26.02 container (needs `bootstrap_main.sh` first).
- ⏸ End-to-end pretrain/finetune not executed on this 1-GPU dev box — requires real image/video data + a 4-8 GPU allocation.

## Related

- Module 03 — same `FSDP2Manager` primitive, different model class.
- Module 04 — LoRA on LLMs. Same `PeftConfig` class; different `target_modules` pattern (module-name glob on transformer layers instead of match-all-linear).
- Module 05 — for pure image generation with a small SD variant, the classic diffusers API works without FSDP sharding.
- `schedulers.md` — sampler / step-count / flow-matching vs DDPM cheat sheet.
- Upstream: https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/diffusion
