# Module 16 — Mamba / state-space models

Automodel's LLM recipe (`train_ft.py`) is **architecture-agnostic**: as long as the model is a `transformers.PreTrainedModel` subclass that takes `input_ids` and returns a loss-bearing output when given `labels`, the recipe runs it unchanged.

That makes training a Mamba-2 state-space model (no attention, sub-quadratic in sequence length) a one-config-change exercise — no custom model code, no recipe changes, no new CLI verb.

## What this module demonstrates

Pretraining a tiny Mamba-2 (~13.5M params) on TinyShakespeare through the *same* `pretrain llm` CLI you used in Module 02. Only the YAML's `model.config._target_` differs:

```diff
-    _target_: transformers.GPT2Config
+    _target_: transformers.Mamba2Config
```

That's the whole story — same `train_ft.py`, same dataset, same FSDP2 sharding, same DCP checkpoints, same HF consolidation output.

## Smoke-verified (on H200, 1 GPU)

5 steps pretrain:
```
step 0 | loss 11.49
step 4 | loss 11.44
Saving checkpoint to .../tiny_mamba2_smoke/epoch_0_step_4
```
Consolidated ckpt loads back as `Mamba2ForCausalLM` via `AutoModelForCausalLM.from_pretrained`.

## Run

```bash
sbatch 16_mamba/mamba_run.slrm                 # 1 GPU, ~5 min
```

Under the hood: same `02_pretrain/pretrain.sh` wrapper, different config.

## Mamba-2 config knobs

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_config
  config:
    _target_: transformers.Mamba2Config
    vocab_size: 50258
    hidden_size: 128
    num_hidden_layers: 6
    num_heads: 8          # Mamba-2 SSM heads (think attention heads)
    head_dim: 32
    expand: 2             # SSM expansion factor
    n_groups: 1
    state_size: 16        # SSM recurrent-state width
```

**SSM constraint:** `hidden_size * expand == num_heads * head_dim`. Here: `128 * 2 == 8 * 32 == 256`. Transformers silently errors at config init if this fails.

## Why Mamba, why here

The workshop flows modules 02 (transformer pretrain) → 11 (custom transformer) → 13 (KD between transformers) → 16 (non-transformer). The progression shows Automodel's abstraction level: it doesn't care whether the model is GPT-2, a custom RoPE transformer, or a state-space model — it cares that the object exposes `forward(input_ids, labels=None) -> CausalLMOutput`.

Concrete upshot: if tomorrow someone ships a new architecture (Hyena, RWKV, BitNet, Titans) with a `PreTrainedModel` wrapper, you can train it through Automodel the same afternoon.

## Gotchas

- **Fast kernels not in this container build.** Mamba's canonical speedup comes from `mamba_ssm.selective_state_update` + `causal_conv1d` CUDA kernels. The container has `mamba_ssm==2.3.0` but a symbol mismatch with HF's `transformers==5.0.0` Mamba-v1 adapter (Mamba-2 is fine but falls back to a pure-PyTorch path). Throughput on this container: ~14k tps vs ~26k tps for tiny GPT-2 at the same size. Correctness is identical; only perf is hurt.
- To enable fast kernels, rebuild the SIF adding `pip install causal-conv1d` and a fresh `pip install mamba-ssm` — see NVIDIA's [build args](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docker/Dockerfile) for the mamba extras pattern.
- **Large pretrained Mamba checkpoints** (`state-spaces/mamba-1.4b-hf`, `AntonV/mamba2-780m`) are on HF Hub — use `NeMoAutoModelForCausalLM.from_pretrained` (Module 04 Track B style) to finetune them instead of pretraining from scratch.
- **Mamba has no attention → no TP plan.** `distributed.tp_size > 1` is unsupported; stick with FSDP2 `dp_size` only. Mamba inherits long-context advantage from its sub-quadratic recurrence — no need for context-parallelism either.
- **KV cache semantics differ.** Inference (Module 05) still works — HF's `generate()` knows how to handle Mamba's recurrent-state cache — but performance characteristics vs a transformer are different at long context (Mamba linear in seq_len, transformer quadratic).

## Finetune a pretrained Mamba (alternative path)

To adapt an existing Mamba checkpoint instead of pretraining a tiny one:

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: AntonV/mamba2-780m   # or state-spaces/mamba-1.4b-hf
  torch_dtype: bf16
```

Then swap this into Module 04's `finetune.sh` call path — LoRA, full-finetune, or KD (Module 13) all work with no extra code.

## Related

- Module 02 — transformer equivalent. Compare tps/loss/memory on the same data.
- Module 11 — registering a *custom* architecture as a `PreTrainedModel`. Mamba is the "don't write code, just use HF's existing wrapper" version of the same pattern.
- Module 13 — KD transformer → Mamba (or vice versa) is a recent research direction; should work by swapping teacher/student model types.
