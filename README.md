# NeMo Automodel

Training examples and cluster diagnostics for Kempner-style H100/H200 Slurm clusters, built on the NeMo-Automodel Singularity image. Two top-level trees: [`basic_examples/`](basic_examples/) (hands-on training modules) and [`cluster_bench/`](cluster_bench/) (HPC health / acceptance / regression suite).

Both trees run inside the same SIF via [`basic_examples/shared/launch.sh`](basic_examples/shared/launch.sh); `cluster_bench/` reuses it through a `shared → ../basic_examples/shared` symlink.

## [`basic_examples/`](basic_examples/) — NeMo-Automodel workshop

18 numbered modules covering the full Automodel workflow end-to-end. Run in order the first time through; each depends on artifacts from the previous.

| #  | Name               | Purpose                                                                        |
|----|--------------------|--------------------------------------------------------------------------------|
| 00 | [`00_setup/`](basic_examples/00_setup/) | Container smoke test: imports, CUDA, FSDP2 config fields.                      |
| 01 | [`01_data/`](basic_examples/01_data/) | Tokenize + shard text for `NanogptDataset`; build chat/SFT JSONL.              |
| 02 | [`02_pretrain/`](basic_examples/02_pretrain/) | Pretrain nanoGPT-style GPT-2 from random init (tiny + FineWeb scales).         |
| 03 | [`03_distributed/`](basic_examples/03_distributed/) | FSDP2 / MegatronFSDP / TP / PP / HSDP; DCP checkpointing; HF export.           |
| 04 | [`04_finetune/`](basic_examples/04_finetune/) | SFT Track A (continue pretrained GPT-2) and Track B (HF model + LoRA).         |
| 05 | [`05_inference/`](basic_examples/05_inference/) | Generation via `AutoModelForCausalLM.generate` (one-shot, REPL, batched).      |
| 06 | [`06_profiling/`](basic_examples/06_profiling/) | nsys via benchmark recipe; `torch.profiler` sidecar; grid-sweep autotuner.     |
| 07 | [`07_eval/`](basic_examples/07_eval/) | `lm-eval-harness` via writable overlay + LLM-as-judge.                         |
| 08 | [`08_fault_tolerance/`](basic_examples/08_fault_tolerance/) | Kill-and-resume, DCP reshardable load, torch elastic, signal-safe saves.    |
| 09 | [`09_fp8/`](basic_examples/09_fp8/) | FP8 training via Transformer Engine + `torch.compile` (H100/H200 only).        |
| 10 | [`10_long_context/`](basic_examples/10_long_context/) | Long-context training via sequence-parallel (SP) and context-parallel (CP).    |
| 11 | [`11_custom_model/`](basic_examples/11_custom_model/) | Register a custom `PreTrainedModel` subclass (RoPE-GPT) via `_target_`.        |
| 12 | [`12_vllm_serve/`](basic_examples/12_vllm_serve/) | vLLM OpenAI-compatible server against a consolidated HF checkpoint.            |
| 13 | [`13_kd/`](basic_examples/13_kd/) | Knowledge distillation: Qwen3-1.7B teacher → Qwen3-0.6B student.               |
| 14 | [`14_scaling/`](basic_examples/14_scaling/) | Scaling ladder 100M → 120B, one new parallelism knob per step.                 |
| 15 | [`15_vlm/`](basic_examples/15_vlm/) | VLM finetune: Gemma3-VL-4B LoRA on CORD-V2 via `finetune vlm` CLI.             |
| 16 | [`16_mamba/`](basic_examples/16_mamba/) | Mamba-2 state-space model via the same LLM recipe (architecture-agnostic).     |
| 17 | [`17_diffusion/`](basic_examples/17_diffusion/) | Diffusion & flow matching: pretrain / finetune (LoRA) / generate.              |

See [`basic_examples/README.md`](basic_examples/README.md) for prerequisites, the launch pattern, and env-var reference.

## [`cluster_bench/`](cluster_bench/) — HPC diagnostic suite

Repurposes the workshop's benchmark recipe and distributed primitives as a cluster health / acceptance / regression suite. Six build stages:

1. **Static diagnostics** — `nvidia-smi` topology snapshots, GPU↔IB-HCA affinity audit, IB counter-delta checker. See [`cluster_bench/scripts/`](cluster_bench/scripts/).
2. **Training baselines** — per-node MFU on Qwen3-1.7B (straggler detection) + world-size scaling on Qwen2.5-7B (intra-node NVLink, inter-node IB). See [`cluster_bench/sbatch/`](cluster_bench/sbatch/).
3. **NCCL microbenchmarks** — writable overlay with nccl-tests; intra-node, inter-node, and all-pairs heatmaps. See [`cluster_bench/nccl_tests/`](cluster_bench/nccl_tests/).
4. **Storage** — DCP checkpoint throughput across filesystem tiers (tmp / netscratch / holylfs).
5. **MTTR + drift** — kill-and-resume wall-clock, 7-day p50 baselines, regression flagging with sparklines.
6. **Acceptance gate** — single [`accept_node.slrm`](cluster_bench/sbatch/accept_node.slrm) runs all seven checks and emits `PASS` / `WARN` / `FAIL`.

Hardcoded for Kempner (`partition=kempner_dev`, `/n/netscratch` + `/n/holylfs06` paths, shared SIF). See [`cluster_bench/README.md`](cluster_bench/README.md) for thresholds, the results layout, and the correlator / report pipeline.

## Start here

- Workshop walkthrough: **[`basic_examples/README.md`](basic_examples/README.md)**
- Cluster diagnostics: **[`cluster_bench/README.md`](cluster_bench/README.md)**
