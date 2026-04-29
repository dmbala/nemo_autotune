# Module 14 — Scaling from 100M to 120B

A ladder of benchmark configs that exercise one new parallelism knob at each step of model scale. Each config runs the `benchmark` recipe from Module 06 — so throughput / MFU / iter-time are measured in a consistent way, and comparing steps on the same hardware tells you exactly where each knob starts paying off.

## The ladder

| # | File | Model | New knob vs previous step | Min world size |
|---|---|---|---|---|
| 1 | (see Module 02) | nanoGPT ~100M | — single GPU, no sharding | 1 GPU |
| 2 | `step2_qwen3_1p7b_fsdp2.yaml` | Qwen3-1.7B | FSDP2 `dp_size` | 4 GPU (1 node) |
| 3 | `step3_qwen25_7b_fsdp2_ac.yaml` | Qwen2.5-7B | `activation_checkpointing: true` | 4 GPU |
| 4 | `step4_qwen25_32b_tp_sp.yaml` | Qwen2.5-32B | `tp_size: 2` + `sequence_parallel: true` | 4 GPU |
| 5 | `step5_qwen3_moe30b_ep.yaml` | Qwen3-MoE-30B-A3B | `ep_size: 8` | 8 GPU (1 node) |
| 6 | `step6_llama33_70b_pp_2nodes.yaml` | Llama-3.3-70B | `pp_size: 2` across nodes | 8 GPU (2 nodes) |
| 7 | `step7_gptoss_120b_multinode.yaml` | GPT-OSS-120B | full stack (EP=64 across 8 nodes) | 64 GPU (8 nodes) |

Each config uses `MockIterableDataset` so no data prep is required — the point is measurement, not learning.

## Decision tree

See [`decision_tree.md`](./decision_tree.md) for the "which knob, when?" flowchart and OOM troubleshooting table. TL;DR:

```
Scale memory first with FSDP2 + activation checkpointing;
scale model width with TP + SP within a node;
scale model depth with PP across nodes;
scale MoE experts with EP;
scale sequence length with CP.
```

## Run a single step

```bash
# Pick a step by name:
sbatch --export=ALL,STEP=step3 14_scaling/ladder.slrm

# Or let the sbatch auto-select based on world size (defaults to --nodes=1 --gres=gpu:4):
sbatch 14_scaling/ladder.slrm

# Multi-node steps (override sbatch directives):
sbatch --nodes=2 --export=ALL,STEP=step6 14_scaling/ladder.slrm
sbatch --nodes=8 --ntasks-per-node=8 --gres=gpu:8 --export=ALL,STEP=step7 14_scaling/ladder.slrm
```

Each run writes `$RESULTS_ROOT/scaling/<step>/benchmark_results.json` with:
```json
{
  "avg_iter_time_seconds": ...,
  "avg_mfu_percent": ...,
  "tflops_per_gpu": ...
}
```

## Collate results into a ladder table

After running a few steps, use Module 06's `report.py` on the scaling results:

```bash
# Copy per-step JSONs into a common index format:
RES=$RESULTS_ROOT/scaling
python3 -c "
import json, pathlib
rows = []
for sd in sorted(pathlib.Path('${RES}').iterdir()):
    p = sd / 'benchmark_results.json'
    if p.exists():
        d = json.loads(p.read_text())
        d['_rid'] = sd.name
        d['_overrides'] = {}
        rows.append(d)
(pathlib.Path('${RES}') / 'index.json').write_text(json.dumps(rows, indent=2))
"
shared/launch.sh python 06_profiling/report.py --results $RESULTS_ROOT/scaling
```

You get a markdown table sorted by MFU%. Expect a roughly monotone decrease in MFU as you go up the ladder — each new communication primitive costs something.

## Honest caveats

- **Steps 1–3 are real.** You can run them on a 4-GPU allocation and measure MFU.
- **Steps 4–5 are reachable on a full H100/H200 node** (8 GPUs) — the Qwen3-30B-A3B model needs `trust_remote_code: true` and a few minutes of HF download.
- **Steps 6–7 are reference configs.** Running them requires real cluster allocations (2 nodes for step 6, 8 nodes for step 7) plus in step 7's case Transformer Engine kernels + DeepEP for the all-to-all expert dispatch. TE 2.11 is in the container; DeepEP isn't — the container was built with `INSTALL_UCCL_EP=False`. The config falls back to `attn: sdpa / linear: torch / dispatcher: torch` which will train but much slower than the upstream `te_deepep` variant at `/opt/Automodel/examples/benchmark/configs/gptoss_120b_te_deepep.yaml`.
- **Model names are HF IDs.** For gated models (Llama-3.3-70B), set `HF_TOKEN` before launching. Qwen3 models are ungated.

## Reading the throughput numbers

Typical MFU on H100 BF16:

| Scale | Expected MFU | What limits it |
|---|---|---|
| 1.7B dense | 35–45 % | Compute-bound at small seq_len; kernel launch overhead at batch=1 |
| 7B dense | 45–55 % | Usually the sweet spot for dense bf16 FSDP2 |
| 30B dense + TP | 35–45 % | TP all-reduce adds per-block communication |
| 30B MoE + EP | 25–35 % | All-to-all expert dispatch is costly without DeepEP |
| 70B + PP | 30–40 % | Pipeline bubbles + inter-node bandwidth |
| 120B MoE, full stack | 25–35 % | Multi-hop comm, sensitive to network topology |

If your numbers are much lower, see Module 06's `sweep.py` to tune `local_batch_size` / `seq_len` / parallelism splits.

## What this module is and isn't

**Is**: a measurement-first ladder for building intuition about where each parallelism primitive starts mattering, anchored in actual Automodel configs.

**Isn't**: a production training guide for 70B/120B models — those need hyperparam tuning, data pipelines, failure recovery (Module 08), evaluation (Module 07), and multi-week compute budgets. Use the steps 1–5 configs as starting points for real runs on your own hardware; treat steps 6–7 as reference implementations to study rather than turn-key training.

## Related

- Module 02 — the 100M starting point on 1 GPU.
- Module 03 — single-node FSDP2 + HSDP + DCP.
- Module 06 — the benchmark recipe + autotuner sweep this ladder builds on.
- Module 09 — FP8 via TE, orthogonal to this ladder (combines with any step).
- Module 10 — long context (CP) for the sequence-length axis.
- `/opt/Automodel/examples/benchmark/configs/` — upstream reference configs for every major model family.
