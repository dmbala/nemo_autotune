# Module 07 — Evaluation + LLM-as-judge

Two complementary eval paths. Automodel ships neither; we install `lm-eval-harness` into a writable Singularity overlay and provide an LLM-as-judge runner.

| Tool | Produces | Good for |
|---|---|---|
| `lm-eval-harness` | Task-level accuracy on MMLU / HellaSwag / ARC-Easy etc. | Standardized benchmarks; side-by-side vs other models. |
| `llm_judge.py` | Rubric scores (1-5 per axis) or pairwise winners. | Open-ended generation quality, SFT gains, preference alignment. |

## One-time setup — build the overlay

Run once on a login node (compute nodes have no internet):

```bash
bash 07_eval/install_overlay.sh
```

This creates `07_eval/overlay.img` (4 GB writable overlay) and pip-installs `lm-eval[hf]==0.4.*` into it. Any subsequent `shared/launch.sh` call with `OVERLAY=07_eval/overlay.img` will see the package.

## lm-eval-harness

```bash
sbatch 07_eval/run_lm_eval.slrm
# Customize:  CKPT=<dir> TASKS=hellaswag,arc_easy,mmlu BATCH_SIZE=16 sbatch ...
```

Under the hood:

```bash
lm_eval --model hf \
  --model_args pretrained=<ckpt>,dtype=bfloat16,trust_remote_code=true \
  --tasks hellaswag,arc_easy \
  --batch_size 16 \
  --output_path $RESULTS_ROOT/lm_eval/<name>.json
```

The JSON contains per-task `acc`, `acc_norm`, stderrs, and (with `--log_samples`) per-example predictions.

Point it at any HF-loadable dir: Module 02/03/04 consolidated checkpoints, or a Hub id like `Qwen/Qwen3-0.6B` for baseline.

## LLM-as-judge

Three stages:

1. **Prompts** — sample N prompts from a HF dataset:
   ```bash
   shared/launch.sh python 07_eval/build_eval_prompts.py \
       --dataset rajpurkar/squad --n 100 \
       --out $RESULTS_ROOT/prompts/squad_100.jsonl
   ```

2. **Responses** — run your model under test:
   ```bash
   shared/launch.sh python 05_inference/batch_infer.py \
       --ckpt <ckpt_dir> \
       --in  $RESULTS_ROOT/prompts/squad_100.jsonl \
       --out $RESULTS_ROOT/outputs/model_under_test.jsonl \
       --use-chat-template --batch-size 8
   ```

3. **Judge** — score with a larger reference model:
   ```bash
   sbatch 07_eval/run_judge.slrm          # uses the latest Track B ckpt
   ```

### Judge backends

Set `JUDGE=` env var:

| Spec | Notes |
|---|---|
| `local:Qwen/Qwen2.5-7B-Instruct` (default) | Pulled from HF Hub; fits one H100/H200 in bf16. ~1 GB/s throughput at batch=1. |
| `openai:gpt-4o-mini` | Needs `OPENAI_API_KEY`. Container has `openai` preinstalled. |
| `anthropic:claude-opus-4-6` | Needs `ANTHROPIC_API_KEY`. Container has `anthropic` preinstalled. |

### Modes

**Rubric** (`--responses`): scores each response on helpfulness / correctness / conciseness (1-5). Prints the mean per axis.

**Pairwise** (`--pairwise --responses-a --responses-b`): A vs B comparison. Runs both orderings and only counts a decisive win when both orderings agree — mitigates position bias.

### Judge-bias caveats

LLM judges have measurable biases: self-preference (judge picks its own model), verbosity bias (preferring longer answers), position bias (preferring whichever side is labeled A). Mitigations:

- Use a judge model distinct from the models under test.
- Pairwise: always evaluate both orderings (this module does it automatically).
- Report absolute scores + disagreement rate, not just win rate.
- Spot-check the judge's rationale field in the output JSONL before trusting aggregates.

## Expected outputs

```
$RESULTS_ROOT/
  prompts/squad_100.jsonl
  outputs/model_under_test.jsonl         # one response per prompt
  lm_eval/<ckpt_name>.json               # lm-eval-harness report
  judge/rubric.jsonl                     # one row per prompt, parsed judge JSON
  judge/pairwise.jsonl                   # for --pairwise runs
```

## Gotchas

- `lm_eval` needs the HF checkpoint on a path readable at eval time — use the consolidated dirs Automodel writes.
- `mmlu` has 57 sub-tasks and takes ~30 min on a 1B model. Use `hellaswag,arc_easy` for a quick smoke-run.
- The local judge (Qwen2.5-7B-Instruct) pulls ~15 GB on first use; make sure `HF_HOME` has space.
- API judges are cheapest for small runs (<500 prompts). The local judge is cheaper at scale because H100 is a sunk cost.
- The judge JSONs are best-effort parsed (`_extract_json`); malformed JSON from the judge ends up with null scores. Check the `raw` field when debugging.
