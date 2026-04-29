"""Interactive stdin REPL: type a prompt, get a completion. Uses the tokenizer's
chat template if it has one (e.g. Qwen3), otherwise treats stdin as raw text.

Usage:
    shared/launch.sh python 05_inference/chat_repl.py --ckpt Qwen/Qwen3-0.6B
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--system", default=None, help="System prompt (ignored if tokenizer has no chat template)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    has_chat = getattr(tok, "chat_template", None) is not None
    history: list[dict] = []
    if has_chat and args.system:
        history.append({"role": "system", "content": args.system})

    print(f"[chat] ready ({type(model).__name__}, chat_template={'yes' if has_chat else 'no'}). Ctrl-D to exit.")
    while True:
        try:
            line = input("user> ").strip()
        except EOFError:
            print()
            return 0
        if not line:
            continue

        if has_chat:
            history.append({"role": "user", "content": line})
            prompt_ids = tok.apply_chat_template(
                history, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")
        else:
            prompt_ids = tok(line, return_tensors="pt").input_ids.to("cuda")

        with torch.no_grad():
            out = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tok.pad_token_id,
            )
        reply = tok.decode(out[0][prompt_ids.shape[-1]:], skip_special_tokens=True)
        print(f"assistant> {reply}\n")
        if has_chat:
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    sys.exit(main())
