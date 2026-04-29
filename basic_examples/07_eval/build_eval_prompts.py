"""Build a prompts JSONL for LLM-as-judge evaluation from a HF dataset.

Default: sample N SQuAD validation examples, format each as an open-ended
question answered from context. Output schema:

    {"id": "...", "prompt": "...", "reference": "..."}

The ``reference`` field holds the ground-truth answer (optional; lets the
judge do reference-grounded rubric scoring).

Usage:
    shared/launch.sh python 07_eval/build_eval_prompts.py \
        --dataset rajpurkar/squad --n 100 \
        --out 07_eval/prompts/squad_100.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _squad_prompt(row: dict) -> dict:
    answers = row.get("answers", {}).get("text", [])
    return {
        "id": row["id"],
        "prompt": (
            f"Context:\n{row['context']}\n\n"
            f"Question: {row['question']}\n\n"
            "Answer concisely using only information from the context."
        ),
        "reference": answers[0] if answers else "",
    }


def _alpaca_prompt(row: dict) -> dict:
    user = row["instruction"] + (("\n\n" + row["input"]) if row.get("input") else "")
    return {
        "id": row.get("id") or row.get("instruction")[:32],
        "prompt": user,
        "reference": row.get("output", ""),
    }


FORMATTERS = {"rajpurkar/squad": _squad_prompt, "tatsu-lab/alpaca": _alpaca_prompt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rajpurkar/squad")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    fmt = FORMATTERS.get(args.dataset)
    if fmt is None:
        raise SystemExit(f"no formatter for {args.dataset}; extend FORMATTERS")

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)

    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i in idx:
            f.write(json.dumps(fmt(ds[i])) + "\n")
    print(f"[prompts] wrote {len(idx)} prompts → {args.out}")


if __name__ == "__main__":
    main()
