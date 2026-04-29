# Module 02 — Pre-training nanoGPT

Trains a GPT-2-style causal LM from random init using Automodel's `train_ft.py` recipe. Pretrain and finetune route to the *same* recipe — the difference is the config (and the data). The `build_gpt2_model` builder in `nemo_automodel.components.models.gpt2` returns a fresh-init HF-compatible GPT-2 model, so a student can read the code easily.

## Two scales

| Config | Model | Data | Wall time |
|---|---|---|---|
| `tiny_gpt2_shakespeare.yaml` | 6L × 128d × 4h, ctx=256, ~10M params | Tiny-Shakespeare | ~5 min on 1×H200 |
| `gpt2_124m_fineweb.yaml` | 12L × 768d × 12h, ctx=1024, ~124M params | FineWeb-500M | ~2–4 hr on 4×H100 |

## Run — tiny path (after Module 01 tiny)

```bash
sbatch 02_pretrain/pretrain_tiny.slrm
# → logs/ws02_pretrain_tiny.<jobid>.out
# → $CKPT_ROOT/tiny_gpt2/
```

Under the hood, `pretrain.sh` resolves paths + invokes:

```bash
shared/launch.sh automodel -c configs/tiny_gpt2_shakespeare.yaml pretrain llm \
    --dataset.file_pattern=$DATA_ROOT/shakespeare/train.bin \
    --checkpoint.checkpoint_dir=$CKPT_ROOT/tiny_gpt2 \
    --nproc-per-node=1
```

## Run — FineWeb-500M path (requires Module 01 fineweb step)

```bash
sbatch 01_data/run_fineweb.slrm      # once, ~1 hr
sbatch 02_pretrain/pretrain_124m.slrm
```

Edit `configs/gpt2_124m_fineweb.yaml` to adjust `max_steps`, `local_batch_size`, or use `activation_checkpointing` as needed for your node.

## CLI-level overrides

Anything in the YAML can be overridden at launch time with `--dotted.path=value`:

```bash
shared/launch.sh automodel -c configs/tiny_gpt2_shakespeare.yaml pretrain llm \
    --step_scheduler.max_steps=1000 \
    --optimizer.lr=5e-4 \
    --distributed.dp_size=1
```

The parser lives at `/opt/Automodel/nemo_automodel/components/config/_arg_parser.py`.

## What the recipe does

1. Builds a `torch.distributed.device_mesh` from the `distributed.*` block. With `strategy: fsdp2` + `dp_size: none`, it auto-expands to `world_size` (1 for tiny, 4 for 124M).
2. Instantiates the model via the `_target_` class-path plus the remaining YAML keys as kwargs.
3. Wraps the model with FSDP2, loads the `NanogptDataset`, trains for `max_steps`, and emits checkpoints every `ckpt_every_steps`.
4. With `checkpoint.save_consolidated: true` + `model_save_format: safetensors`, rank-0 also writes an HF-compatible consolidated `model.safetensors` + `config.json` alongside the DCP shards. Module 05 inference loads the consolidated dir.

## Expected outputs

```
$CKPT_ROOT/tiny_gpt2/
  epoch_0_step_<N>/
    config.yaml
    losses.json
    step_scheduler.pt
    rng/  dataloader/  optim/                  # resume-state
    model/
      shard-00001-model-00001-of-00001.safetensors
      consolidated/                            # ← HF-loadable
        config.json
        generation_config.json
        model-00001-of-00001.safetensors
        model.safetensors.index.json
```

Point Module 05 inference and Module 07 eval at the `consolidated/` directory.
If you omit `save_consolidated: true`, use Module 03's `export_to_hf.py` on the
sharded `model/*.safetensors` to produce the same flat HF layout.

> **Why HF-backed model instead of `build_gpt2_model`?** The builder returns a minimal `nn.Module` which doesn't implement `save_pretrained`, so the recipe's consolidation step fails on it. Using `NeMoAutoModelForCausalLM.from_config` with `transformers.GPT2Config` gives you a real HF `GPT2LMHeadModel` that consolidates cleanly.

## Verification

Check that training loss drops:

```bash
grep -E "loss=" logs/ws02_pretrain_tiny.*.out | tail -5
```

Expect something like `loss=10.8 ... loss=3.5` over 500 steps. If it plateaus near 10.8, check that the `file_pattern` resolved to a real `.bin` and that tokens aren't all zeros (rare shard corruption).
