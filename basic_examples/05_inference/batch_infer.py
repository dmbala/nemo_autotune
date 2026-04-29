"""Batched generation over a JSONL of prompts. Writes a sibling JSONL of outputs.

Input  schema (prompts.jsonl): one ``{"id": "...", "prompt": "..."}`` per line.
Output schema (outputs.jsonl): one ``{"id": "...", "prompt": "...", "response": "..."}`` per line.

Used by Module 07 eval + LLM-judge pipelines.

Usage:
    shared/launch.sh python 05_inference/batch_infer.py \
        --ckpt $CKPT_ROOT/trackB_qwen3_0p6b_lora_squad/epoch_0_step_299/model/consolidated \
        --in  07_eval/prompts/eval_prompts.jsonl \
        --out 07_eval/outputs/qwen3_squad.jsonl --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--use-chat-template", action="store_true")
    args = ap.parse_args()

    prompts = [json.loads(l) for l in args.inp.read_text().splitlines() if l.strip()]
    print(f"[batch] {len(prompts)} prompts from {args.inp}")

    tok = AutoTokenizer.from_pretrained(args.ckpt, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out.open("w")
    n = 0
    for batch in _chunks(prompts, args.batch_size):
        if args.use_chat_template and getattr(tok, "chat_template", None):
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": p["prompt"]}],
                    add_generation_prompt=True, tokenize=False,
                )
                for p in batch
            ]
        else:
            texts = [p["prompt"] for p in batch]

        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tok.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        replies = tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)

        for p, r in zip(batch, replies):
            fout.write(json.dumps({"id": p.get("id"), "prompt": p["prompt"], "response": r}) + "\n")
            n += 1
    fout.close()
    print(f"[batch] wrote {n} completions → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
