# Module 04 — Fine-tuning

Two tracks demonstrate the two canonical SFT shapes:

| Track | Starting point | Approach | GPUs | Wall time |
|---|---|---|---|---|
| **A** | Tiny GPT-2 from Module 02 pretrain | Full-model SFT on SQuAD | 1 | ~10 min |
| **B** | Qwen3-0.6B pulled from HF Hub | LoRA adapter SFT on SQuAD | 4 (FSDP2) | ~20–40 min |

Both tracks route through the same `train_ft.py` recipe — the CLI verb (`finetune`) doesn't change the code path, only the defaults. The only real difference from Module 02 pretrain is that `model._target_` is `NeMoAutoModelForCausalLM.from_pretrained` (weights loaded) rather than `from_config` (random init).

## Track A — continue the nanoGPT pretrain

Requires Module 02 tiny pretrain to have completed:

```bash
sbatch 04_finetune/finetune_trackA.slrm
```

The sbatch script discovers the latest `epoch_*_step_*/model/consolidated/` dir under `$CKPT_ROOT/tiny_gpt2`, passes it as `--model.pretrained_model_name_or_path=...`, and SFTs on 2k SQuAD examples. Output: `$CKPT_ROOT/tiny_sft/epoch_0_step_<N>/model/consolidated/`.

## Track B — HF Hub model through the pipeline

Ungated default (Qwen3-0.6B):

```bash
sbatch 04_finetune/finetune_trackB.slrm
```

Gated alternative (Llama-3.2-1B) — needs a HF token:

```bash
export HF_TOKEN=hf_...  # paste your token
CONFIG=llama32_1b_lora_squad.yaml sbatch 04_finetune/finetune_trackB.slrm
```

### Pre-warming the HF cache (optional)

If compute nodes don't have internet, download once on a login node:

```bash
shared/launch.sh python 04_finetune/download_hf_model.py --model Qwen/Qwen3-0.6B
# or: HF_TOKEN=hf_... shared/launch.sh python 04_finetune/download_hf_model.py --model meta-llama/Llama-3.2-1B
```

`HF_HOME` must point at a path readable by compute nodes (default `$SCRATCH_ROOT/.hf`).

## LoRA specifics

Track B uses `nemo_automodel.components._peft.lora.PeftConfig`:

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: true   # attach LoRA to every nn.Linear
  dim: 16                  # rank
  alpha: 32                # scale
  use_triton: true         # triton-autotune'd LoRA kernel
```

With `checkpoint.is_peft: true`, only the adapter weights are saved (a few MB), not the full base model. This is what you want for experiment tracking.

## Expected outputs

Same layout as Module 02/03:
```
$CKPT_ROOT/tiny_sft/epoch_0_step_<N>/
  model/consolidated/        # HF-loadable (full weights for Track A, base+adapter for Track B)
  optim/   rng/   dataloader/
```

For Track B with `is_peft: true`, the consolidated directory contains the **merged** base+LoRA weights via `ConsolidatedHFAddon`. To inspect adapter-only weights, look at `model/shard-*.safetensors`.

## Verification

Track A — eval loss should drop below the Module 02 final loss:

```bash
grep -E 'val.*loss' logs/ws04_trackA_sft.*.out | tail
```

Track B — LoRA adapter trained, merged checkpoint loads:

```bash
shared/launch.sh python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
path = '$CKPT_ROOT/trackB_qwen3_0p6b_lora_squad/epoch_0_step_299/model/consolidated'
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype='bfloat16').to('cuda')
tok = AutoTokenizer.from_pretrained(path)
print(tok.decode(m.generate(**tok('Context: The Nile is the longest river. Question: Which river is the longest?', return_tensors='pt').to('cuda'), max_new_tokens=20)[0]))
"
```

## Gotchas

- Track A's `pretrained_model_name_or_path` is a *local path*, not a Hub ID. Automodel happily accepts either (it's `transformers.AutoModel.from_pretrained` under the hood).
- Llama-3.2 is gated on HF. Without `HF_TOKEN`, the job fails at tokenizer download with a 401. Use the Qwen3 config if you don't have access.
- LoRA with `match_all_linear: true` attaches to more layers than the more common `["q_proj", "v_proj"]` pattern. Adjust if you want narrower adaptation.
- For bigger models (>7B), switch to `distributed.tp_size=2` and require 8 GPUs — see `/opt/Automodel/examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml`.
