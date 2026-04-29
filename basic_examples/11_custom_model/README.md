# Module 11 — Registering a custom model

Automodel's YAML parser is Hydra-like: any `_target_: <dotted.path.to.callable>` gets dynamically imported and called with the remaining keys as kwargs. That applies uniformly to models, datasets, optimizers, schedulers, losses. So dropping in your own model family is a 3-step pattern:

1. Write a `transformers.PreTrainedModel` subclass + matching `PretrainedConfig`.
2. Put the module on `PYTHONPATH`.
3. Point a training YAML at your config class via `NeMoAutoModelForCausalLM.from_config`.

## Why `PreTrainedModel`, not plain `nn.Module`?

Automodel's recipe (`base_recipe.py:353`) unconditionally calls `model.save_pretrained(...)` to write consolidated HF-format checkpoints. A plain `nn.Module` doesn't have that method and the training job will crash at the first checkpoint boundary with `AttributeError: ... has no attribute 'save_pretrained'`.

Wrapping the module in a `PreTrainedModel` buys three things:
- `save_pretrained` for Automodel's consolidation path.
- `from_pretrained` for Module 05 inference (`AutoModelForCausalLM.from_pretrained(<consolidated>)`).
- Registration via `AutoConfig.register(...)` / `AutoModelForCausalLM.register(...)` so downstream code only needs `import custom_model` for HF to recognize the type.

## What's in this module

```
11_custom_model/
├── custom_model.py                  # RoPEGPTConfig + RoPEGPTForCausalLM, registered at import time
├── configs/
│   └── custom_rope_shakespeare.yaml # from_config → RoPEGPTConfig
├── custom_run.slrm                # exports PYTHONPATH, reuses 02_pretrain/pretrain.sh
└── README.md
```

The custom model is a minimal **RoPE-GPT**: rotary positional embeddings, tied LM-head weights, GELU MLP. ~170 lines including the HF wrapper. The architecture is deliberately simple so the registration mechanism — not the model — is the point.

## Run it

```bash
sbatch 11_custom_model/custom_run.slrm
```

The sbatch:
```bash
export PYTHONPATH=$WORKSHOP_ROOT/11_custom_model:$PYTHONPATH
02_pretrain/pretrain.sh configs/custom_rope_shakespeare.yaml \
    $DATA_ROOT/shakespeare/train.bin \
    $CKPT_ROOT/custom_rope_gpt --nproc-per-node=1
```

`shared/launch.sh` forwards `PYTHONPATH` into the Singularity container so `import custom_model` works inside.

## The YAML hook

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_config
  config:
    _target_: custom_model.RoPEGPTConfig   # ← your config class
    vocab_size: 50258
    n_positions: 512
    n_embd: 192
    n_layer: 6
    n_head: 4
    bos_token_id: 50256
    eos_token_id: 50256
```

Automodel's config loader does `importlib.import_module('custom_model')` (which triggers the `AutoConfig.register`/`AutoModelForCausalLM.register` calls at the bottom of the file), instantiates `RoPEGPTConfig(vocab_size=..., ...)`, then calls `NeMoAutoModelForCausalLM.from_config(config)` to build the model.

## Inspection commands

Import + build outside the recipe:

```bash
export PYTHONPATH=$(pwd)/11_custom_model
shared/launch.sh python -c "
import custom_model
import torch
from custom_model import RoPEGPTConfig, RoPEGPTForCausalLM
m = RoPEGPTForCausalLM(RoPEGPTConfig(vocab_size=50258, n_positions=512, n_embd=192, n_layer=6, n_head=4))
x = torch.randint(0, 50258, (2, 32))
out = m(input_ids=x)
print('params:', sum(p.numel() for p in m.parameters())/1e6, 'M')
print('logits:', tuple(out.logits.shape))
"
```

Expected: `params: ~22 M`, `logits: (2, 32, 50258)`.

Load a saved checkpoint back:

```bash
shared/launch.sh python -c "
import custom_model                                        # triggers registration
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('$CKPT_ROOT/custom_rope_gpt/epoch_0_step_499/model/consolidated')
print(type(m).__name__)
"
```

(Without the `import custom_model`, HF raises `ValueError: model type 'rope_gpt' not recognized`.)

## Smoke-verified

- 5-step pretrain on TinyShakespeare: loss 10.82 → 10.54 ✅
- Consolidated ckpt written with `config.json` + `model-00001-of-00001.safetensors` ✅
- Reload via `AutoModelForCausalLM.from_pretrained(...)` → `RoPEGPTForCausalLM, 21.96M params, logits shape (1, 16, 50258)` ✅

## Interface requirements

For `train_ft.py` to happily train your model:

| Requirement | Why |
|---|---|
| Subclass `PreTrainedModel` with a `config_class` | save_pretrained / load_pretrained + HF Auto* registration |
| `forward(input_ids, labels=None, **kwargs)` returning `CausalLMOutput(loss, logits)` | Recipe computes loss against `labels`; accepts HF output type |
| Tie embeddings *before* `post_init()` if using `wte == lm_head.weight` | FSDP2 wrap happens after init |
| Handle arbitrary kwargs (`attention_mask`, `position_ids`, `past_key_values`) via `**kwargs` | The recipe / generate() pass extras; absorb or ignore them |

## Gotchas

- **`PYTHONPATH` must be exported before `launch.sh`.** The wrapper forwards it to the container via `--env`, but only if it's set in the outer shell. The sbatch handles this; for interactive `salloc`, export it manually.
- **Numeric-prefixed directories (`11_custom_model/`) aren't importable as Python packages** (can't start with a digit). That's why the YAML says `custom_model.RoPEGPTConfig` (just the file) and we put the dir on PYTHONPATH rather than importing `workshop.11_custom_model.custom_model`.
- **Registration happens on import.** `custom_model.py` ends with `AutoConfig.register("rope_gpt", ...)` — any process that needs to load the custom ckpt must `import custom_model` first, otherwise `AutoModelForCausalLM.from_pretrained(...)` raises `model type 'rope_gpt' not recognized`.
- **`torch_dtype` kwarg.** Transformers 5.x warns that `torch_dtype` is deprecated in favor of `dtype`. Our config passes it through `PretrainedConfig.__init__` so it works either way.
- **`post_init()` runs `_init_weights`** on every submodule. If you override `_init_weights`, make sure it handles all the module types your model uses (Linear, Embedding, LayerNorm) — missing types get the framework default.

## When to actually do this

Use the custom-model hook for:
- A new attention variant (MoE, sparse attn, linear attn, Mamba-style).
- Novel positional embeddings (RoPE variants, ALiBi, NoPE).
- Architectural experiments (weight-tied shared blocks, deep-narrow exotica).
- Reproducing a research paper without forking Automodel.

Don't use it when the stock HF Transformers implementation is good enough — in that case `NeMoAutoModelForCausalLM.from_pretrained("org/model")` (Module 04 Track B) is zero extra work.
