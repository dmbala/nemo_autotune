# Module 05 — Inference

Automodel is not involved at inference time — the checkpoints it writes are plain HF `save_pretrained` directories, so regular `transformers.AutoModelForCausalLM` works. This module keeps the code footprint small for that reason.

## Scripts

| Script | Use |
|---|---|
| `generate.py` | One-shot completion from CLI args. |
| `chat_repl.py` | Interactive stdin loop. Uses the tokenizer's chat template if present. |
| `batch_infer.py` | Batched generation over a JSONL of prompts (feeds Module 07 eval). |

## Single-prompt generation

```bash
sbatch 05_inference/inference.slrm    # uses the latest tiny_sft/ checkpoint

# Or directly:
shared/launch.sh python 05_inference/generate.py \
    --ckpt $CKPT_ROOT/tiny_sft/epoch_0_step_<N>/model/consolidated \
    --prompt "Context: ... Question: ..." --max-new-tokens 80
```

For a **pretrain-from-scratch** checkpoint (Module 02), there's no tokenizer_config.json in the consolidated dir — pass `--tokenizer gpt2` so we use the GPT-2 BPE you trained against.

## Interactive chat

```bash
shared/launch.sh python 05_inference/chat_repl.py --ckpt Qwen/Qwen3-0.6B
# or against a LoRA-merged ckpt:
shared/launch.sh python 05_inference/chat_repl.py \
    --ckpt $CKPT_ROOT/trackB_qwen3_0p6b_lora_squad/epoch_0_step_299/model/consolidated \
    --system "You are a concise QA assistant."
```

The REPL uses `tokenizer.apply_chat_template` when the tokenizer has one (Qwen, Llama-Instruct, etc.). For bare GPT-2 it falls back to raw-text prompting.

## Batched inference

Input JSONL (`{"id": ..., "prompt": ...}` per line) → output JSONL (`{"id", "prompt", "response"}`):

```bash
shared/launch.sh python 05_inference/batch_infer.py \
    --ckpt <hf_dir> \
    --in 07_eval/prompts/eval_prompts.jsonl \
    --out 07_eval/outputs/model_under_test.jsonl \
    --batch-size 8 --use-chat-template
```

Module 07's `llm_judge.py` reads the output JSONL.

## Gotchas

- **Left padding** matters for batched generation: `AutoTokenizer.from_pretrained(..., padding_side='left')`. Already handled in `batch_infer.py`.
- Our consolidated GPT-2 pretrain dirs don't include a tokenizer — that's fine; use `--tokenizer gpt2` or bundle one yourself via `AutoTokenizer.from_pretrained('gpt2').save_pretrained(ckpt_dir)`.
- On H100/H200, bf16 is fine. For T4/A10, switch to fp16.
- For high-throughput serving, use **vLLM** or **TGI** instead of `model.generate` — that's out of scope for this workshop.
