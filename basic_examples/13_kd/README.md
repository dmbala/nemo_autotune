# Module 13 — Knowledge distillation

Distill a larger **teacher** model into a smaller **student** by matching the teacher's soft output distribution on every token. The student gets closer to teacher-quality than if you had SFT'd it on the same labels alone, and you get to ship the smaller model.

Automodel's `kd` command is the fourth of its four CLI verbs — we've now used all of them:

| CLI verb | Recipe | Used in |
|---|---|---|
| `pretrain` | `train_ft.py` | 02, 03, 08, 11 |
| `finetune` | `train_ft.py` | 04, 09, 10 |
| `benchmark` | `benchmark.py` | 06 |
| **`kd`** | **`kd.py`** | **13 (this module)** |

## The loss

From `/opt/Automodel/nemo_automodel/recipes/llm/kd.py`:

```
loss = (1 - kd_ratio) * ce_loss  +  kd_ratio * kd_loss
```

- `ce_loss` — masked cross-entropy against ground-truth labels (same as SFT).
- `kd_loss` — KL divergence between student and teacher distributions, both soft-max'd at `temperature T`.
  - Higher `T` flattens distributions, forcing the student to attend to low-probability tokens (richer signal but harder to optimize).
  - `T=1`: hard distillation, closest to pure CE on teacher argmax.
  - `T=2–4`: typical sweet spot.

A `kd_ratio: 0.0` run degenerates to plain SFT (Module 04 Track A); `1.0` ignores labels entirely and just mimics the teacher.

## Requirements

- **Matching tokenizer.** Student and teacher must produce the same vocab, or the logit alignment breaks. Pick same-family models (Qwen3 + Qwen3, Llama + Llama). Cross-family distillation needs an extra projection layer the recipe doesn't handle.
- **GPU budget**: teacher sits resident on each rank, in `eval()` + `no_grad`. Memory cost ≈ student + teacher (no optimizer state for teacher). For our 0.6B student + 1.7B teacher in bf16, ≈ 4.5 GB combined parameter memory → comfortably fits 4 × H100.
- **HF_HOME** pointing at a path readable by all ranks (both models get pulled).

## Configs

| Config | Student | Teacher | kd_ratio | Notes |
|---|---|---|---|---|
| `kd_qwen3_student0p6b_teacher1p7b.yaml` | Qwen3-0.6B (full) | Qwen3-1.7B | 0.5 | Balanced teacher + label signal |
| `kd_qwen3_lora_student0p6b.yaml` | Qwen3-0.6B + LoRA(r=16) | Qwen3-1.7B | 0.7 | Cheap adaptation, lean harder on teacher |

## Run

```bash
# Full-finetune KD (default):
sbatch 13_kd/kd_run.slrm

# LoRA-KD variant:
CONFIG=kd_qwen3_lora_student0p6b.yaml sbatch 13_kd/kd_run.slrm
```

Under the hood:
```bash
shared/launch.sh automodel -c configs/<cfg>.yaml kd llm \
    --checkpoint.checkpoint_dir=$CKPT_ROOT/... \
    --nproc-per-node=4
```

Note the **`kd llm`** command (not `finetune llm`) — this routes to the KD recipe which knows about the `teacher_model` and `kd_loss_fn` blocks.

## Knobs worth tuning

| Key | Effect |
|---|---|
| `kd_ratio` | 0.0 → SFT; 1.0 → pure teacher mimicry. Start at 0.5, raise if the student plateaus short of the teacher. |
| `kd_loss_fn.temperature` | 1 (hard) to 4 (soft). Higher T picks up more of the teacher's lower-rank token preferences. |
| `kd_loss_fn.fp32_upcast` | Safer KL numerics; slight slowdown. Leave `true`. |
| `peft` (LoRA) | Turn on when you want cheap adaptation or to distill into a frozen-backbone student. |

## Verification

```bash
# 1. Confirm KD-specific logging (loss components split in the log):
grep -E 'ce_loss|kd_loss|kd_ratio' logs/ws13_kd.*.out | head

# 2. Student ckpt is a standard HF dir (no teacher stored):
ls $CKPT_ROOT/kd_kd_qwen3_student0p6b_teacher1p7b/epoch_0_step_299/model/consolidated/
#  → config.json  model-*.safetensors  generation_config.json

# 3. Side-by-side with pure SFT: run Module 04 Track B for the same 300 steps and
#    compare validation losses. Good KD runs show a noticeable gap (student+teacher
#    below student-only) within 100 steps.

# 4. Judge it (Module 07):
MODEL=$CKPT_ROOT/kd_.../epoch_0_step_299/model/consolidated \
    sbatch 07_eval/run_judge.slrm
```

## Gotchas

- **Tokenizer mismatch is silent and catastrophic.** If you point `teacher_model.pretrained_model_name_or_path` at a model with a different vocabulary than the student, loss will train toward random. Same family only.
- **Teacher in bf16 is usually fine.** If you see unstable KD loss (sudden spikes), try `teacher_model.torch_dtype: fp32` — the extra precision on teacher logits smooths the soft distribution.
- **`kd_ratio` schedules** aren't supported out of the box. If you want to anneal (e.g., 1.0 → 0.0 over training), patch the recipe or stage two runs with different ratios.
- **Don't save the teacher.** `checkpoint.is_peft: true` stores only LoRA adapters; even without it, the recipe skips the teacher. The consolidated dir has student weights only — reload with `AutoModelForCausalLM.from_pretrained(...)` normally.
- **`peft` + KD** LoRA-adapts only the *student*. The teacher is always fully frozen. `is_peft: true` keeps the saved adapter tiny.

## Related

- Module 04 — plain SFT; useful as a baseline to compare KD against.
- Module 07 — score the KD'd student with `lm-eval-harness` or the LLM-as-judge.
- `/opt/Automodel/examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml` — upstream example (Llama-3.2-1B student + Llama-3.2-3B teacher).
- [Hinton et al. 2015, "Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531) — the original KD paper.
