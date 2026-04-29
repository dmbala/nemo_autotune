# Module 09 — FP8 training with Transformer Engine

Eight-bit floating-point training via NVIDIA Transformer Engine gets you ~1.5–2× throughput on H100/H200 vs bfloat16, with negligible accuracy loss when the recipe is set up right. Automodel wires this in by (a) calling TE's linear/attention under the hood when `fp8.enabled: true` and (b) letting `torch.compile` fuse the FP8 kernels.

## Requirements

- H100 / H200 (native FP8). Pre-Hopper GPUs support `fp8.emulate: true` for correctness testing only — zero perf win.
- Transformer Engine installed. **Already present in `nemo-automodel-26.02-fixed.sif`** (`transformer_engine 2.11.0`). Verified:
  ```bash
  shared/launch.sh python -c "import transformer_engine as te; print(te.__version__)"
  ```

## What the config does

The key additions over a normal LoRA finetune (Module 04 Track B):

```yaml
compile:
  enabled: true
  mode: default            # fuse kernels; use 'max-autotune' for final runs
  dynamic: false

fp8:
  enabled: true
  recipe_name: tensorwise  # one scale per tensor. See "Recipes" below.
  enable_fsdp_float8_all_gather: true
  precompute_float8_dynamic_scale_for_fsdp: true
  force_recompute_fp8_weight_in_bwd: true
  filter_fqns: ["lm_head"] # keep output head in higher precision
  emulate: false
```

## Recipes

FP8 "recipe" = scaling strategy. Automodel passes through to TE:

| Recipe | Scale granularity | Accuracy | Speed |
|---|---|---|---|
| `tensorwise` | one scale per tensor | OK for most finetunes | fastest |
| `rowwise` | one scale per row of the matmul | better for large-vocab / LM heads | small overhead |
| `rowwise_with_gw_hp` | rowwise + keep gradient-weight in high precision | safest | slight perf hit |

Start with `tensorwise`; escalate if you see loss divergence vs a bf16 run.

## Run

```bash
sbatch 09_fp8/fp8_run.slrm
```

Under the hood:
```bash
04_finetune/finetune.sh \
    09_fp8/configs/fp8_qwen3_0p6b_lora.yaml \
    $CKPT_ROOT/fp8_qwen3_lora \
    --nproc-per-node=4
```

Same recipe (`train_ft.py`) as Modules 02/04 — only the config differs.

## Verify FP8 is actually active

Look for TE initialization in the log:

```bash
grep -E 'TE|fp8|Float8|DelayedScaling' logs/ws09_fp8.*.out | head
```

Expected lines include `TE: compiling model with fp8 recipe ...` and TE's warmup scale logs.

Sanity check the first few steps against a bf16 baseline (drop `fp8:` block and rerun): loss curves should track within noise. A divergence of >10% vs bf16 at step ~100 means the recipe is too aggressive — try `rowwise_with_gw_hp`.

## Expected throughput

For a 0.6B LoRA finetune on 4× H100, expect roughly:

| Precision | TFLOPs/GPU | tokens/s/GPU |
|---|---|---|
| bf16 | ~200 | ~20k |
| fp8 (tensorwise) | ~350 | ~35k |

(Use Module 06 `sweep.py` with `fp8.enabled` toggled on/off to measure on your hardware.)

## Gotchas

- **Always** keep `filter_fqns: ["lm_head"]`. LM heads have large fan-out and FP8 noise there blows up loss.
- `emulate: true` is bf16 under the hood — throughput is *worse* than plain bf16. Only for correctness testing on older GPUs.
- `enable_fsdp_float8_all_gather` requires the FP8 weights to be sharded in FP8 during all-gather, which saves bandwidth. If you get a shape error in FSDP2 init, disable it first — it's the most fragile knob.
- `torch.compile` + FP8 + LoRA + PEFT has historically been touchy. If compile fails with a cryptic error, set `compile.enabled: false` first and re-enable after confirming the non-compiled FP8 path works.
- For the benchmark recipe (Module 06), FP8 reduces iter time but `peak_tflops=989` is the BF16 number; MFU% will look inflated. Use tokens/s as the apples-to-apples metric.
