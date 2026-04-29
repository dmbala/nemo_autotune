# Module 15 — Vision-language model finetuning

Automodel's second CLI domain is `vlm`. The recipe `/opt/Automodel/nemo_automodel/recipes/vlm/finetune.py` parallels `train_ft.py` in shape but:

1. Loads the model via `NeMoAutoModelForImageTextToText.from_pretrained(...)` instead of `NeMoAutoModelForCausalLM`.
2. Uses VLM-specific datasets and collators under `nemo_automodel.components.datasets.vlm.*`.
3. Supports a `freeze_config` block to freeze the vision tower / embeddings while LoRA-adapting the text decoder — the canonical low-cost VLM finetune.

## What's in this module

```
15_vlm/
├── configs/
│   └── gemma3_vl_4b_cord_lora.yaml   # Gemma3-VL-4B LoRA on CORD-V2 (receipt parsing)
├── vlm_finetune.sh                    # launches automodel ... finetune vlm
├── vlm_run.slrm
└── README.md
```

The default config finetunes **Gemma3-VL-4B** (ungated, 4B params incl. ViT) on **CORD-V2** (~1k receipts → structured JSON). LoRA adapters on text-side linears; vision tower frozen. Fits 4×H100 in bf16.

## Run

```bash
sbatch 15_vlm/vlm_run.slrm
```

Under the hood:
```bash
shared/launch.sh automodel -c configs/gemma3_vl_4b_cord_lora.yaml finetune vlm \
    --checkpoint.checkpoint_dir=$CKPT_ROOT/vlm_gemma3_cord_lora \
    --nproc-per-node=4
```

Note the **`finetune vlm`** — the domain switch routes to `recipes/vlm/finetune.py`.

## VLM-specific YAML

Compared to an LLM config, the new pieces are:

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained   # not CausalLM
  pretrained_model_name_or_path: google/gemma-3-4b-it
  torch_dtype: torch.bfloat16

peft:
  match_all_linear: False
  exclude_modules:                     # don't LoRA the vision side
    - "*vision_tower*"
    - "*vision*"
    - "*visual*"
    - "*image_encoder*"
    - "*lm_head*"

dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset
  path_or_dataset: naver-clova-ix/cord-v2

dataloader:
  collate_fn:
    _target_: nemo_automodel.components.datasets.vlm.collate_fns.default_collate_fn

freeze_config:
  freeze_embeddings: true
  freeze_vision_tower: true
  freeze_language_model: false
```

## Other VLM examples upstream

If CORD-V2 isn't your target, Automodel ships configs for:

| Model | Dataset | Domain | Path |
|---|---|---|---|
| Gemma3-VL-4B | MedPIX (medical imaging) | radiology | `examples/vlm_finetune/gemma3/gemma3_vl_4b_medpix{,_peft}.yaml` |
| Gemma3n-VL-4B | MedPIX | radiology | `examples/vlm_finetune/gemma3n/*.yaml` |
| InternVL-3.5-4B | CORD-V2 | doc QA | `examples/vlm_finetune/internvl/internvl_3_5_4b.yaml` |
| Kimi-VL 2.5/2 | CORD-V2 / MedPIX | doc / radiology | `examples/vlm_finetune/kimi/*.yaml` |

To use one, copy the YAML into `configs/` and swap the `CONFIG_PATH` in the sbatch. Gated models (if any) need `HF_TOKEN`.

## Inference

For generation from a VLM checkpoint, use `transformers.AutoModelForImageTextToText` directly (analog to Module 05's `generate.py` but with an image input):

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

processor = AutoProcessor.from_pretrained(ckpt_dir)
model = AutoModelForImageTextToText.from_pretrained(ckpt_dir, torch_dtype="bfloat16").to("cuda")
inputs = processor(images=Image.open("receipt.jpg"), text="Extract the total.", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=200)
print(processor.batch_decode(out, skip_special_tokens=True))
```

## Gotchas

- **`torch_dtype: torch.bfloat16`** (dotted string, not `bf16`) in VLM configs — this matches the upstream VLM examples. The LLM-side configs in this workshop use `bf16` (a shorter alias). Both work; just don't mix within one YAML.
- **`match_all_linear: False` + `exclude_modules`** is the inverse of the LLM pattern. VLMs have thousands of matmuls across the vision tower that you don't want to adapt — opt-in exclusion is safer than opt-in inclusion.
- **CORD-V2 downloads 1.2 GB on first use.** Warm the HF cache with `download_hf_model.py` (Module 04) using `--model naver-clova-ix/cord-v2` if you need to avoid network on the compute node.
- **`attn_implementation: eager`** is the upstream recommendation for Gemma3-VL. Flash-attn paths have been flaky with the vision cross-attention in recent transformers releases.
- **No VLM benchmark recipe.** Module 06's throughput-sweep ladder is LLM-only. For VLMs, profile with `torch.profiler` (Module 06's sidecar pattern) or just time epochs.

## Related

- Module 04 — LLM LoRA finetune (same PEFT primitives, different modality).
- Module 05 — inference patterns that generalize to VLMs with `AutoModelForImageTextToText`.
- Module 07 — LLM-as-judge can score VLM outputs too (feed the model's answer as text; the judge is blind to the image).
