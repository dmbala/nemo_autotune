"""Convert a HuggingFace instruction dataset (e.g. ``rajpurkar/squad``,
``tatsu-lab/alpaca``) into the JSONL chat schema accepted by
``nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset``.

Output JSONL — one example per line:

    {"messages": [
        {"role": "user", "content": "<prompt>"},
        {"role": "assistant", "content": "<response>"}
    ]}

Used by Module 04 (Track B).

Usage:
    shared/launch.sh python 01_data/jsonl_to_chat.py \
        --dataset rajpurkar/squad --split train \
        --out data/sft/squad_train.jsonl --max-samples 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_squad(row: dict) -> dict:
    answers = row.get("answers", {}).get("text", [])
    answer = answers[0] if answers else ""
    user = f"Context:\n{row['context']}\n\nQuestion: {row['question']}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]
    }


def _format_alpaca(row: dict) -> dict:
    instr = row["instruction"]
    inp = row.get("input", "") or ""
    user = f"{instr}\n\n{inp}".strip()
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["output"]},
        ]
    }


FORMATTERS = {
    "rajpurkar/squad": _format_squad,
    "tatsu-lab/alpaca": _format_alpaca,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rajpurkar/squad")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    fmt = FORMATTERS.get(args.dataset)
    if fmt is None:
        raise SystemExit(
            f"No formatter for {args.dataset}. Add one to FORMATTERS in this script."
        )

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as f:
        for row in ds:
            f.write(json.dumps(fmt(row)) + "\n")
            n += 1
    print(f"[chat] wrote {n} examples → {args.out}")


if __name__ == "__main__":
    main()
