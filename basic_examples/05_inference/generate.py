"""Single-prompt generation from any HF-format checkpoint.

Works on any directory produced by Automodel's ``save_consolidated=true`` path
(Module 02 / 03 / 04) or any HF Hub id.

Usage:
    shared/launch.sh python 05_inference/generate.py \
        --ckpt $CKPT_ROOT/tiny_sft/epoch_0_step_<N>/model/consolidated \
        --prompt "Context: The Nile is the longest river. Question: Which river is the longest?"
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Local dir or HF Hub id")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--tokenizer", default=None,
                    help="If the ckpt lacks a tokenizer (pretrain-from-scratch), pass e.g. 'gpt2'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    print(f"[generate] model: {type(model).__name__}  params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    inputs = tok(args.prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tok.pad_token_id,
    )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    print(f"\n=== prompt ===\n{args.prompt}\n\n=== completion ===\n{decoded[len(args.prompt):]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
