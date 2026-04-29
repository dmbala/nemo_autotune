# NeMo-Automodel Workshop

Hands-on modules teaching the full NeMo-Automodel workflow end-to-end on Kempner-style SLURM HPC (H100/H200 nodes, 4 GPUs/node) with a Singularity-packaged container.

## Prerequisites

- Access to the Kempner cluster and a Slurm account (default: `kempner_dev`).
- Read access to the prebuilt container:
  `/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/nemo/nemo-automodel-26.02-fixed.sif`
  (This is the `-fixed` variant — see [Container gotcha](#container-gotcha)).
- A writable netscratch path (default: `/n/netscratch/kempner_dev/Lab/$USER/Agent/nemo/runs`).

## Module map

| # | Name | What you learn |
|---|------|----------------|
| 00 | `00_setup/` | Container usage; smoke-test imports + CUDA. |
| 01 | `01_data/` | Tokenize + shard text data for `NanogptDataset`; build chat/SFT JSONL. |
| 02 | `02_pretrain/` | Pretrain a nanoGPT-style GPT-2 from random init on tiny + FineWeb scales. |
| 03 | `03_distributed/` | FSDP2 / MegatronFSDP / TP / PP / HSDP; DCP distributed checkpointing; export to HF. |
| 04 | `04_finetune/` | SFT track A (continue from our pretrained GPT-2) and track B (download a real HF model → LoRA). |
| 05 | `05_inference/` | Generation via `AutoModelForCausalLM.generate` (single prompt, chat REPL, batched). |
| 06 | `06_profiling/` | nsys profiling via the benchmark recipe; `torch.profiler` sidecar; grid-sweep autotuner. |
| 07 | `07_eval/` | `lm-eval-harness` via writable overlay + LLM-as-judge (local Qwen2.5-7B or API). |
| 08 | `08_fault_tolerance/` | Kill-and-resume via `restore_from: LATEST`; DCP reshardable load; torch elastic; signal-safe saves. |
| 09 | `09_fp8/` | FP8 training via Transformer Engine + torch.compile. H100/H200 only. |
| 10 | `10_long_context/` | Long context via sequence-parallel (SP) and context-parallel (CP) configs. |
| 11 | `11_custom_model/` | Registering your own `PreTrainedModel` subclass (RoPE-GPT) via `_target_`. |
| 12 | `12_vllm_serve/` | vLLM OpenAI-compatible server against a consolidated HF checkpoint (overlay install). |
| 13 | `13_kd/` | Knowledge distillation — Qwen3-1.7B teacher → Qwen3-0.6B student via the `kd` CLI command. |
| 14 | `14_scaling/` | Scaling ladder 100M → 120B; one new parallelism knob per step, all measured with the benchmark recipe. |
| 15 | `15_vlm/` | VLM finetune — Gemma3-VL-4B LoRA on CORD-V2 via the `finetune vlm` CLI. |
| 16 | `16_mamba/` | Mamba-2 (state-space model) pretrained via the same LLM recipe — architecture-agnostic demo. |
| 17 | `17_diffusion/` | Diffusion & flow-matching — pretrain + finetune (LoRA) + distributed generate. Flux, Wan2.1/2.2, Hunyuan. Needs `bootstrap_main.sh` (train recipe added post-26.02). |

Run modules in order the first time through — each depends on artifacts from the previous one (data → ckpt → export → ft → inference → eval).

## The launch pattern

Everything runs inside the container. `shared/launch.sh` is the single entry point:

```bash
# Activate venv inside the container and run anything:
shared/launch.sh automodel -c 02_pretrain/configs/tiny_gpt2_shakespeare.yaml pretrain llm
shared/launch.sh python 00_setup/smoke_test.py
```

On compute nodes, `SBATCH` scripts source `shared/slurm_common.sh` for standard env (`MASTER_ADDR`, `MASTER_PORT`, `HF_HOME`, paths, NCCL hygiene).

### Environment variables

| Var | Meaning | Default |
|-----|---------|---------|
| `SIF` | path to the Singularity image | shared fixed SIF |
| `HF_HOME` | HF Hub cache root (models + tokenizers land here) | `$SCRATCH_ROOT/.hf` |
| `HF_TOKEN` | required for gated models (e.g. Llama) | unset |
| `SCRATCH_ROOT` | where runs / ckpts / results land | `/n/netscratch/kempner_dev/Lab/$USER/Agent/nemo/runs` |
| `OVERLAY` | path to a writable overlay image | unset (only used in 07) |

## Container gotcha

The stock `nvcr.io/nvidia/nemo-automodel:26.02` image bakes `/opt/Automodel/nemo_automodel` as `770 root:root`, which blocks non-root users from importing the package under plain `singularity exec`. The `-fixed` SIF in this workshop was rebuilt with the trivial overlay below:

```singularity
Bootstrap: localimage
From: /path/to/nemo-automodel-26.02.sif

%post
    chmod -R a+rX /opt/Automodel /opt/venv
```

If you ever need to rebuild the fix, see `shared/user_overlay.sh` and the `nemo-automodel-fixed.def` in the repo parent directory.

## What Automodel ships (the key facts)

- CLI: `automodel -c <yaml> <finetune|pretrain|kd|benchmark> <LLM|VLM>`. `finetune` and `pretrain` both dispatch to `nemo_automodel/recipes/llm/train_ft.py` — the config decides.
- From-scratch GPT-2 pretraining is first-class (`nemo_automodel.components.models.gpt2.build_gpt2_model`).
- Distributed: FSDP2, MegatronFSDP, DDP, TP/PP/CP/EP, HSDP. Configured under the YAML `distributed.*` block.
- Checkpoints: `torch.distributed.checkpoint` (DCP) with safetensors sharding, async save (torch≥2.9), consolidated HF export for inference.
- Profiling: `nsys` only inside the benchmark recipe. No built-in `torch.profiler` — Module 06 supplies a user-land sidecar.
- **Not** shipped: RL/RLHF recipes, `lm-eval-harness`, `evaluate`, LLM-as-judge tooling. Module 07 builds a writable overlay for eval. RL is explicitly out of scope (use TRL/verl).

## Quick start

```bash
cd /n/netscratch/kempner_dev/Lab/$USER/Agent/nemo/workshop

# 1. Smoke test (1 GPU, ~1 min)
sbatch 00_setup/smoke_test.slrm

# 2. Prepare tiny data (login node, ~30 s)
shared/launch.sh python 01_data/tiny_shakespeare.py

# 3. Tiny pretrain (1 GPU, ~5 min)
sbatch 02_pretrain/pretrain_tiny.slrm

# ... continue through modules 03–07.
```

See each module's `README.md` for details and verification steps.
