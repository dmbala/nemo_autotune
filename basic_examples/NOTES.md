# NeMo-Automodel Workshop — Plan & Build Notes

Reference doc for picking this up again later. Combines the original plan, the decisions made, what was built, what was smoke-verified, and what's still pending.

---

## 1. Context

A hands-on, modular workshop teaching the full NeMo-Automodel workflow end-to-end on a Kempner-style SLURM HPC (H100/H200 nodes, 4 GPUs/node), using a nanoGPT-scale model as the didactic vehicle and a real HF-Hub model (Qwen3-0.6B) as the production-shape example.

**Container**: `/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/nemo/nemo-automodel-26.02-fixed.sif` (9.8 GB). This is the `chmod a+rX /opt/Automodel /opt/venv` overlay over NVIDIA's stock NGC `nvcr.io/nvidia/nemo-automodel:26.02` — the stock image ships `/opt/Automodel/nemo_automodel` as `770 root:root`, which blocks non-root imports under plain `singularity exec`. Build recipe lives at `../nemo-automodel-fixed.def`.

**Workshop root**: `/n/netscratch/kempner_dev/Lab/$USER/Agent/nemo/workshop/`.

---

## 2. What Automodel actually ships (key facts)

Findings from reading `/opt/Automodel/*` during build. These shaped every module.

| Feature | Present? | Location |
|---|---|---|
| CLI: `automodel -c <yaml> {finetune,pretrain,kd,benchmark} {llm,vlm}` | yes | `nemo_automodel/_cli/app.py` |
| `finetune` and `pretrain` share recipe | yes | both → `nemo_automodel/recipes/llm/train_ft.py` |
| From-scratch nanoGPT pretraining | yes | `examples/llm_pretrain/nanogpt_pretrain.yaml` + `components/models/gpt2.py::build_gpt2_model` |
| FSDP2 + MegatronFSDP + DDP + TP + PP + CP + EP + HSDP | yes | `components/distributed/config.py`, parsed by `recipes/_dist_setup.py` |
| DCP distributed checkpointing (safetensors sharded) | yes | `components/checkpoint/checkpointing.py` (`dcp.save` L667, `dcp.load` L622) |
| Async DCP save (torch ≥ 2.9) + consolidated HF export | yes | `CheckpointingConfig.is_async`, `save_consolidated` |
| HF Hub model loading | yes | `NeMoAutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=...)` |
| LoRA / PEFT | yes | `nemo_automodel.components._peft.lora.PeftConfig` (`match_all_linear`, `dim`, `alpha`, `use_triton`) |
| Benchmark recipe (MFU / TFLOPs / iter_time JSON) | yes | `recipes/llm/benchmark.py::BenchmarkingRecipeForNextTokenPrediction` |
| nsys profiling hooks | yes (only these) | `benchmark.{nsys_start, nsys_end, nsys_ranks}` in the benchmark recipe |
| RL recipes (DPO/GRPO/PPO/RLHF) | **no** | — |
| Autoconfigurator / autotune | **no** | — |
| `torch.profiler` integration | **no** | user-land sidecar in Module 06 |
| `lm-eval-harness` / `evaluate` | **not installed** | overlay install in Module 07 |
| LLM-as-judge | **no** | custom script in Module 07 |

The `GPT2LMHeadModel` returned by `build_gpt2_model` is a plain `nn.Module` without `save_pretrained`, so the recipe's consolidation path fails on it. Workshop pretrain configs use `NeMoAutoModelForCausalLM.from_config` with `transformers.GPT2Config` instead — gives a real HF `GPT2LMHeadModel`.

---

## 3. User-facing design decisions

| Topic | Decision | Why |
|---|---|---|
| RL module | **skipped** | No recipes in Automodel; covering it properly needs TRL/verl + its own module |
| Autotuner | **grid sweep via benchmark recipe** | closest primitive Automodel ships; composable |
| Model scale | **both** | ~10M GPT-2 on TinyShakespeare (laptop-scale) + 124M GPT-2 on FineWeb-500M (single-node 4-GPU FSDP2) |
| Real HF model | **Qwen3-0.6B** default, Llama-3.2-1B alternative | ungated, small; Automodel already ships `qwen3_0p6b_hellaswag_peft.yaml` as a reference |
| Delivery format | **scripts + READMEs** (no notebooks) | matches how HPC users actually work |
| Node shape | **H100/H200, 4 GPUs/node** | user-confirmed |

---

## 4. Tree layout

```
workshop/
├── README.md                        # top-level
├── NOTES.md                         # this file
├── .refs/                           # cat'd reference files from /opt/Automodel (not versioned)
├── shared/
│   ├── launch.sh                    # singularity exec + venv source + CA-bundle fix
│   ├── slurm_common.sh              # SBATCH preamble: paths, MASTER_ADDR, NCCL
│   └── user_overlay.sh              # helper to create writable overlay.img
├── 00_setup/
│   ├── smoke_test.py                # versions, cuda matmul, FSDP2 fields
│   ├── smoke_test.slrm
│   └── README.md
├── 01_data/
│   ├── tiny_shakespeare.py          # TinyShakespeare → .bin + .bos.idx (NanogptDataset format)
│   ├── fineweb_10bt.sh              # wraps /opt/Automodel/tools/nanogpt_data_processor.py
│   ├── jsonl_to_chat.py             # SQuAD/Alpaca → ChatDataset JSONL
│   ├── run_fineweb.slrm
│   └── README.md
├── 02_pretrain/
│   ├── configs/
│   │   ├── tiny_gpt2_shakespeare.yaml   # 6L×128d×4h, ctx=256, ~10M params
│   │   └── gpt2_124m_fineweb.yaml       # 12L×768d×12h, ctx=1024, 124M params, FSDP2 dp=4
│   ├── pretrain.sh
│   ├── pretrain_tiny.slrm
│   ├── pretrain_124m.slrm
│   └── README.md
├── 03_distributed/
│   ├── configs/
│   │   ├── fsdp2_dp4.yaml
│   │   ├── fsdp2_multinode_2x4.yaml     # HSDP: dp_replicate_size=2, dp_size=4
│   │   └── ckpt_async_consolidated.yaml # is_async=true, save_consolidated=true
│   ├── export_to_hf.py                  # DCP shards → HF save_pretrained layout
│   ├── run_fsdp2_single.slrm
│   ├── run_hsdp_multinode.slrm        # srun torchrun c10d rendezvous
│   └── README.md
├── 04_finetune/
│   ├── configs/
│   │   ├── tiny_sft_squad.yaml          # Track A: continue our pretrained tiny GPT-2
│   │   ├── qwen3_0p6b_lora_squad.yaml   # Track B: HF download + LoRA (default)
│   │   └── llama32_1b_lora_squad.yaml   # Track B alt (gated; HF_TOKEN)
│   ├── download_hf_model.py             # login-node cache warmer
│   ├── finetune.sh
│   ├── finetune_trackA.slrm
│   ├── finetune_trackB.slrm
│   └── README.md
├── 05_inference/
│   ├── generate.py                  # single-prompt, temp/top_k/top_p
│   ├── chat_repl.py                 # interactive, uses chat template if present
│   ├── batch_infer.py               # prompts.jsonl → outputs.jsonl
│   ├── inference.slrm
│   └── README.md
├── 06_profiling/
│   ├── configs/
│   │   ├── benchmark_base.yaml          # base for sweep driver
│   │   └── nsys_gpt2_124m.yaml          # benchmark.nsys_* hooks
│   ├── run_nsys.slrm                  # nsys profile -c cudaProfilerApi
│   ├── torch_profiler_patch.py          # standalone sidecar (Automodel doesn't wire it)
│   ├── sweep.py                         # grid over (batch, seq_len, dp×tp) → benchmark_results.json
│   ├── report.py                        # leaderboard by MFU%
│   ├── sweep.slrm
│   ├── view_trace.md
│   └── README.md
├── 07_eval/
│   ├── install_overlay.sh               # build overlay.img + pip install lm-eval[hf]
│   ├── run_lm_eval.sh
│   ├── run_lm_eval.slrm
│   ├── build_eval_prompts.py            # HF dataset → prompts.jsonl
│   ├── judge_prompts.yaml               # rubric + pairwise templates
│   ├── llm_judge.py                     # local Qwen2.5-7B + openai + anthropic backends
│   ├── run_judge.slrm
│   └── README.md
├── 08_fault_tolerance/                  # added later — see below
│   ├── configs/tiny_resume.yaml         # restore_from: LATEST + frequent ckpts
│   ├── kill_and_resume.{sh,sbatch}      # phase 1 waits for first ckpt, SIGKILLs, phase 2 resumes
│   ├── reshard_demo.{sh,sbatch}         # save dp=4 → load dp=2
│   ├── chaos_monkey.py                  # per-rank self-destruct wrapper for torch elastic
│   ├── elastic_restart.slrm           # torchrun --max-restarts=3
│   ├── signal_demo.md                   # DistributedSignalHandler lib exists but recipe doesn't use it
│   └── README.md
├── 09_fp8/                              # FP8 training via Transformer Engine
│   ├── configs/fp8_qwen3_0p6b_lora.yaml # fp8: enabled; compile: enabled; recipe_name: tensorwise
│   ├── fp8_run.slrm
│   └── README.md
├── 10_long_context/                     # sequence- and context-parallel demos
│   ├── configs/long_ctx_sp.yaml         # tp=2, dp=2, sequence_parallel=true, seq_len=8192
│   ├── configs/long_ctx_cp.yaml         # dp=2, cp=2, seq_len=16384
│   ├── long_ctx.slrm
│   └── README.md
├── 11_custom_model/                     # register your own PreTrainedModel
│   ├── custom_model.py                  # RoPEGPTConfig + RoPEGPTForCausalLM (HF-registered)
│   ├── configs/custom_rope_shakespeare.yaml
│   ├── custom_run.slrm
│   └── README.md
├── 12_vllm_serve/                       # OpenAI-compatible server
│   ├── install_overlay.sh               # overlay.img with vLLM 0.11.x
│   ├── serve.sh                         # vllm serve wrapper
│   ├── client_example.py                # openai SDK → localhost:8000/v1
│   ├── serve.slrm
│   └── README.md
├── 13_kd/                               # Knowledge distillation — fourth CLI verb
│   ├── configs/
│   │   ├── kd_qwen3_student0p6b_teacher1p7b.yaml   # full-finetune student
│   │   └── kd_qwen3_lora_student0p6b.yaml          # LoRA-adapted student
│   ├── kd.sh                            # automodel CLI wrapper (kd llm)
│   ├── kd_run.slrm
│   └── README.md
├── 14_scaling/                          # Scaling ladder 100M → 120B
│   ├── configs/
│   │   ├── step2_qwen3_1p7b_fsdp2.yaml         # + FSDP2
│   │   ├── step3_qwen25_7b_fsdp2_ac.yaml       # + activation_checkpointing
│   │   ├── step4_qwen25_32b_tp_sp.yaml         # + TP + SP
│   │   ├── step5_qwen3_moe30b_ep.yaml          # + EP
│   │   ├── step6_llama33_70b_pp_2nodes.yaml    # + PP across 2 nodes
│   │   └── step7_gptoss_120b_multinode.yaml    # full stack, 8 nodes
│   ├── decision_tree.md                 # when-to-use flowchart + OOM triage
│   ├── ladder.sh                        # runs one step through the benchmark recipe
│   ├── ladder.slrm                    # auto-picks step based on world size
│   └── README.md
├── 15_vlm/                              # Vision-language finetuning (finetune vlm domain)
│   ├── configs/gemma3_vl_4b_cord_lora.yaml     # Gemma3-VL-4B LoRA, vision tower frozen
│   ├── vlm_finetune.sh
│   ├── vlm_run.slrm
│   └── README.md
├── 16_mamba/                            # State-space model via same LLM recipe
│   ├── configs/tiny_mamba2_shakespeare.yaml    # Mamba-2 with transformers.Mamba2Config
│   ├── mamba_run.slrm
│   └── README.md
└── 17_diffusion/                        # Diffusion & flow matching (pretrain + finetune + generate)
    ├── configs/                         # flow-matching training YAMLs copied from upstream main
    │   ├── pretrain_flux_flow.yaml
    │   ├── pretrain_wan2_1_t2v_flow.yaml
    │   ├── finetune_flux_lora.yaml
    │   ├── finetune_wan2_1_t2v_lora.yaml
    │   └── finetune_wan2_1_t2v_multinode.yaml
    ├── bootstrap_main.sh                # clones upstream main → shadow-mounts /opt/Automodel
    ├── pretrain.py / pretrain.sh / pretrain.slrm    # TrainDiffusionRecipe launchers
    ├── finetune.py / finetune.sh / finetune.slrm
    ├── generate.py / generate.slrm    # NeMoAutoDiffusionPipeline + FSDP2 DiT sharding (works on 26.02)
    ├── schedulers.md                    # DDIM/Euler/flow-matching step-count cheat sheet
    └── README.md
```

**Important container caveat:** pretrain + finetune require the `nemo_automodel.recipes.diffusion.train.TrainDiffusionRecipe` module, which was added *after* the 26.02 container tag. `bootstrap_main.sh` clones main-branch Automodel locally and the wrapper scripts shadow-mount it over `/opt/Automodel` via Singularity `--bind`. Generation works on the stock 26.02 container without bootstrap.

---

## 5. Smoke-verified (on holygpu8a10301, H200, 1 GPU)

1. Module 00 smoke: imports + bf16 matmul + `FSDP2Config` fields listed.
2. Module 01 tiny data: TinyShakespeare → `train.bin` (321,124 tokens) + `.bos.idx`; `NanogptDataset` round-trips (`{'input_ids': [...], 'labels': [...]}`).
3. Module 02 tiny pretrain: 30 training steps, loss 10.76 → 9.33, consolidated HF safetensors saved. `AutoModelForCausalLM.from_pretrained(<consolidated>)` loads as `GPT2LMHeadModel` with 7.66M params.
4. Module 04 Track A: loaded the tiny pretrain checkpoint, ran 5 SFT steps on SQuAD, loss 9.14 → 8.92, consolidated ckpt saved to `tiny_sft_smoke/epoch_0_step_4/model/consolidated/`.
5. Module 05 inference: loaded the 30-step pretrain ckpt via HF, generated 20 tokens from "To be or not to be" — output is gibberish (as expected), pipeline works.
6. Module 08 kill-and-resume: trained tiny GPT-2 to first checkpoint (step_99), killed the process group, relaunched → resumed from step_299 (LATEST at kill time), continued at step 300 with loss 6.54 vs fresh-init 10.80 → optimizer + model state correctly restored.
7. Module 11 custom model: pretrained the RoPE-GPT via recipe (loss 10.82 → 10.54 over 5 steps), consolidated ckpt emitted, reloaded via `AutoModelForCausalLM.from_pretrained(...)` as `RoPEGPTForCausalLM` (21.96M params).

---

## 6. Not yet smoke-verified (needs multi-GPU / HF auth / longer allocs)

Config shapes match extracted reference YAMLs but have not been run end-to-end on this session's 1-GPU node:

- Module 01 FineWeb processing (`01_data/fineweb_10bt.sh`) — needs 32 CPUs, ~1 hr.
- Module 02 124M run (4-GPU FSDP2) — needs `--gres=gpu:4`.
- Module 03 distributed configs + multi-node HSDP — needs 1 or 2 nodes × 4 GPUs.
- Module 04 Track B Qwen3-0.6B / Llama-3.2-1B — needs 4 GPUs + HF download.
- Module 06 nsys + sweep — needs 4-GPU alloc + nsys CLI.
- Module 07 `lm-eval-harness` (needs `install_overlay.sh` on login node first) + LLM-as-judge (needs Qwen2.5-7B download, ~15 GB).
- Module 08 reshard (4 GPUs → 2 GPUs) and elastic_restart (4 GPUs + torchrun --max-restarts) — configs are structurally verified but not end-to-end run on this dev machine.
- Module 09 FP8 — TE 2.11.0 is installed and importable; full 4-GPU FP8 finetune run not executed on this dev box (1 GPU only).
- Module 10 long-context SP/CP — configs structurally verified; memory/throughput curves not measured on this box.
- Module 12 vLLM — overlay install not executed yet (requires login-node with internet + disk for ~8 GB overlay); serve.sh pattern structurally correct per vLLM 0.11.x docs.
- Module 13 KD — YAML schema load-verified (teacher_model / kd_ratio / kd_loss_fn resolve correctly). Full 4-GPU run needs both Qwen3-0.6B and Qwen3-1.7B in HF cache; end-to-end not executed on this 1-GPU dev box.
- Module 14 scaling ladder — all 6 step YAMLs schema-load-verified. Parallelism footprints: step2 baseline / step3 + AC / step4 dp=2+tp=2+SP / step5 ep=8 / step6 dp=2+tp=2+pp=2 / step7 ep=64. Steps 2–4 reachable on a single 4-GPU alloc; step 5 needs 8 GPUs; steps 6–7 need 2 and 8 nodes respectively.
- Module 15 VLM — schema-load-verified (`NeMoAutoModelForImageTextToText.from_pretrained`, `freeze_config`, CORD-V2 dataset builder). Full 4-GPU Gemma3-VL-4B LoRA run not executed on this 1-GPU dev box.
- Module 16 Mamba — smoke-verified: 5-step Mamba-2 pretrain on TinyShakespeare, loss 11.49 → 11.44, consolidated ckpt saved. Falls back to pure-PyTorch Mamba2 kernels (~14k tps vs ~26k for tiny GPT-2).
- Module 17 diffusion — generation deps importable (`diffusers 0.35.2`, `NeMoAutoDiffusionPipeline`, `FSDP2Manager`, `AutoencoderKLWan`) on the 26.02 container as-is. `TrainDiffusionRecipe` was added to Automodel *after* the 26.02 tag — `bootstrap_main.sh` clones main and `--bind`s it over `/opt/Automodel`. Upstream pretrain.py/finetune.py copied verbatim (29-line launchers). Configs (Flux + Wan2.1-T2V, flow-matching + LoRA + multinode) copied from upstream. End-to-end training not executed — needs the bootstrap + real image/video data + 4–8 GPU alloc.

When resuming, start with: `sbatch 02_pretrain/pretrain_tiny.slrm` (already verified with a 30-step smoke run; full 500 steps untested) then walk through the modules in order on a cluster allocation.

---

## 7. Known gotchas (all documented in per-module READMEs)

- **`automodel` domain must be lowercase** (`llm`/`vlm`). Discovered on first pretrain run.
- **`build_gpt2_model` doesn't have `save_pretrained`** — breaks `save_consolidated=true`. Workshop pretrain configs use `NeMoAutoModelForCausalLM.from_config` with `transformers.GPT2Config` instead.
- **Host CA-bundle env leaks into container** — `SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt` on the host doesn't exist in the Ubuntu-based container. `shared/launch.sh` overrides to `/etc/ssl/certs/ca-certificates.crt`.
- **`/opt/Automodel/nemo_automodel` is 770 root:root in the stock NGC image.** Always use the `-fixed` SIF (or `--fakeroot`).
- **`dp_size: none`** is a string meaning "auto", not Python None. Resolved from world_size by `recipes/_dist_setup.py`.
- **HSDP sizing constraint:** `dp_replicate_size × dp_size × tp_size × pp_size × cp_size × ep_size == world_size`. Mismatches crash at mesh setup.
- **Async DCP requires torch ≥ 2.9.** The container has 2.10.0a0, so it works; older torch silently downgrades to sync.
- **Multi-node launches use explicit `torchrun`**, not `automodel` CLI (CLI doesn't pass `--node-rank` per task).
- **Benchmark recipe infers `vocab_size` via `AutoConfig.from_pretrained`** of `cfg.model.config.pretrained_model_name_or_path`. Custom random-init models without a Hub id don't go through this path — use a real HF model in benchmark configs.
- **`peak_tflops`** in `benchmark.*` must match your hardware (H100/H200=989, A100=312). MFU is computed relative to this.
- **Empty `lm_eval` / `evaluate`** in the container — build `07_eval/overlay.img` once via `install_overlay.sh` on a login node, then run with `OVERLAY=07_eval/overlay.img`.
- **Judge self-preference & position bias**: pairwise mode automatically evaluates both orderings and only counts decisive wins when both agree.

---

## 8. Critical upstream files (for future grepping)

Read these when extending the workshop:

- `/opt/Automodel/nemo_automodel/_cli/app.py` — CLI dispatch, `COMMAND_ALIASES`, Slurm submit path.
- `/opt/Automodel/nemo_automodel/recipes/llm/train_ft.py` — pretrain+finetune recipe (L378, L562 HF loading; L143 DeviceMesh; L1489 main).
- `/opt/Automodel/nemo_automodel/recipes/llm/benchmark.py` — benchmark recipe (L49 knobs, L437 JSON dump, L294 cudaProfilerStop).
- `/opt/Automodel/nemo_automodel/recipes/_dist_setup.py` — YAML → dataclass parser.
- `/opt/Automodel/nemo_automodel/components/distributed/config.py` — `FSDP2Config` / `MegatronFSDPConfig` / `DDPConfig` / `PipelineConfig`.
- `/opt/Automodel/nemo_automodel/components/checkpoint/checkpointing.py` — `CheckpointingConfig` (L99), `dcp.save` (L667), `dcp.load` (L622), `ConsolidatedHFAddon` (L123).
- `/opt/Automodel/nemo_automodel/components/models/gpt2.py` — `build_gpt2_model` (note: minimal nn.Module, no `save_pretrained`).
- `/opt/Automodel/nemo_automodel/components/datasets/llm/nanogpt_dataset.py` — `.bin` magic/version/layout, yields `{"input_ids": [...], "labels": [...]}`.
- `/opt/Automodel/tools/nanogpt_data_processor.py` — HF → `.bin` shard pipeline + writer class.
- `/opt/Automodel/examples/llm_pretrain/nanogpt_pretrain.yaml` — our pretrain starting point.
- `/opt/Automodel/examples/llm_finetune/qwen/qwen3_0p6b_hellaswag_peft.yaml` — Track B LoRA template.
- `/opt/Automodel/examples/llm_finetune/gemma/gemma_3_270m_squad.yaml` — smallest production finetune example (useful to crib from).
- `/opt/Automodel/examples/benchmark/configs/qwen3_moe_30b_torch.yaml` — benchmark YAML reference.

Quick extractor when you need any of these locally:

```bash
SIF=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/nemo/nemo-automodel-26.02-fixed.sif
singularity exec "$SIF" cat /opt/Automodel/<path> > workshop/.refs/<basename>
```

---

## 9. Running order for a clean walkthrough

Assumes a GPU allocation with 4 GPUs. Adjust sbatch paths if running from a shell.

```bash
cd /n/netscratch/kempner_dev/Lab/$USER/Agent/nemo/workshop

# 00 — smoke
sbatch 00_setup/smoke_test.slrm

# 01 — tiny data first (fast); kick off fineweb in parallel
shared/launch.sh python 01_data/tiny_shakespeare.py --out $SCRATCH_ROOT/data/shakespeare
sbatch 01_data/run_fineweb.slrm                   # ~1 hr; needed by 02_124m / 03 / 06

# 02 — pretrain
sbatch 02_pretrain/pretrain_tiny.slrm             # ~5 min, 1 GPU
sbatch 02_pretrain/pretrain_124m.slrm             # ~2–4 hr, 4 GPUs; after fineweb

# 03 — distributed
sbatch 03_distributed/run_fsdp2_single.slrm                                   # 4 GPUs
CONFIG=ckpt_async_consolidated.yaml sbatch 03_distributed/run_fsdp2_single.slrm
sbatch 03_distributed/run_hsdp_multinode.slrm                                 # 2 nodes

# 04 — finetune
sbatch 04_finetune/finetune_trackA.slrm                                       # after 02_tiny
sbatch 04_finetune/finetune_trackB.slrm                                       # 4 GPUs
# or gated:
#   HF_TOKEN=hf_... CONFIG=llama32_1b_lora_squad.yaml sbatch 04_finetune/finetune_trackB.slrm

# 05 — inference
sbatch 05_inference/inference.slrm                # generates from latest tiny_sft ckpt

# 06 — profiling
sbatch 06_profiling/run_nsys.slrm
sbatch 06_profiling/sweep.slrm                    # ~30 min

# 07 — eval (one-time overlay build on a login node)
bash 07_eval/install_overlay.sh                     # ~10 min
sbatch 07_eval/run_lm_eval.slrm                   # hellaswag, arc_easy
sbatch 07_eval/run_judge.slrm                     # rubric with local Qwen2.5-7B

# 08 — fault tolerance
sbatch 08_fault_tolerance/kill_and_resume.slrm    # 1 GPU, ~2 min — smoke-verified
sbatch 08_fault_tolerance/reshard_demo.slrm       # 4 GPUs
sbatch 08_fault_tolerance/elastic_restart.slrm    # 4 GPUs, torchrun --max-restarts=3

# 09 — FP8 via Transformer Engine (preinstalled in the SIF)
sbatch 09_fp8/fp8_run.slrm                        # 4 GPUs, Qwen3-0.6B LoRA + FP8

# 10 — long context
MODE=sp sbatch 10_long_context/long_ctx.slrm      # tp=2 + SP, seq_len 8k
MODE=cp sbatch 10_long_context/long_ctx.slrm      # cp=2,      seq_len 16k

# 11 — custom model (smoke-verified)
sbatch 11_custom_model/custom_run.slrm            # 1 GPU, RoPE-GPT on Shakespeare

# 12 — vLLM serve
bash   12_vllm_serve/install_overlay.sh             # once on a login node (~10–20 min)
sbatch 12_vllm_serve/serve.slrm                   # 1 GPU, serves Qwen3-0.6B + fires one client query

# 13 — knowledge distillation (fourth CLI verb)
sbatch 13_kd/kd_run.slrm                          # 4 GPUs, Qwen3-0.6B student + Qwen3-1.7B teacher on SQuAD
CONFIG=kd_qwen3_lora_student0p6b.yaml sbatch 13_kd/kd_run.slrm   # LoRA-student variant

# 14 — scaling ladder (one knob per step, measured via benchmark recipe)
sbatch                        --export=ALL,STEP=step2 14_scaling/ladder.slrm   # 4 GPUs, Qwen3-1.7B
sbatch                        --export=ALL,STEP=step3 14_scaling/ladder.slrm   # 4 GPUs, Qwen2.5-7B + AC
sbatch                        --export=ALL,STEP=step4 14_scaling/ladder.slrm   # 4 GPUs, Qwen2.5-32B + TP + SP
sbatch --gres=gpu:8           --export=ALL,STEP=step5 14_scaling/ladder.slrm   # 8 GPUs, Qwen3-MoE-30B + EP
sbatch --nodes=2              --export=ALL,STEP=step6 14_scaling/ladder.slrm   # 2×4 GPUs, Llama-3.3-70B + PP
sbatch --nodes=8 --gres=gpu:8 --export=ALL,STEP=step7 14_scaling/ladder.slrm   # 8×8 GPUs, GPT-OSS-120B full stack

# 15 — VLM finetune
sbatch 15_vlm/vlm_run.slrm                        # 4 GPUs, Gemma3-VL-4B LoRA on CORD-V2

# 16 — Mamba pretrain (state-space model via same LLM recipe)
sbatch 16_mamba/mamba_run.slrm                    # 1 GPU, tiny Mamba-2 on TinyShakespeare

# 17 — Diffusion & flow-matching
# Generation (works on 26.02 container as-is):
sbatch 17_diffusion/generate.slrm                    # 4 GPUs, Wan2.2-T2V-A14B
PROMPT="..." NUM_STEPS=12 sbatch 17_diffusion/generate.slrm

# Training (needs main-branch recipe — bootstrap first):
bash   17_diffusion/bootstrap_main.sh                  # once, clones main → _automodel_main
sbatch 17_diffusion/finetune.slrm                    # 4 GPUs, Wan2.1-T2V LoRA (flow matching)
CONFIG=finetune_flux_lora.yaml sbatch 17_diffusion/finetune.slrm
sbatch 17_diffusion/pretrain.slrm                    # 8 GPUs, Wan2.1-T2V from scratch
```

---

## 10. Open follow-ups

- Validate the full 500-step tiny pretrain matches training curves (currently smoke-verified to 30 steps only).
- Verify Track B Qwen3-0.6B LoRA finetune runs on a 4-GPU allocation.
- Exercise the `export_to_hf.py` path (runs that set `save_consolidated: false`).
- Dial in `peak_tflops` for H200 specifically (currently 989, same as H100 BF16 dense; H200 is similar peak but different memory BW — MFU numbers will be close but not identical).
- Consider adding a ninth module on RL (TRL DPO) as a follow-up workshop if users ask.
- Tiny GPT-2 pretrain-from-scratch checkpoints don't include a tokenizer in the consolidated dir. Module 05 currently handles this via `--tokenizer gpt2`; could instead auto-bundle `AutoTokenizer.from_pretrained('gpt2').save_pretrained(ckpt_dir)` at end of Module 02.
- Module 08's `reshard_demo` and `elastic_restart` need a 4-GPU allocation to smoke-verify; the kill_and_resume flow is smoke-tested.
- Consider patching `train_ft.py` to wire `DistributedSignalHandler` around the training loop so SIGTERM (SLURM preemption) saves a final checkpoint rather than losing the last `ckpt_every_steps` worth of progress. ~15 lines; would be valuable upstream.
