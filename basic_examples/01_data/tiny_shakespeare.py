"""Download Tiny-Shakespeare and emit a `.bin` + `.bos.idx` pair readable by
``nemo_automodel.components.datasets.llm.nanogpt_dataset.NanogptDataset``.

The binary layout matches `/opt/Automodel/tools/nanogpt_data_processor.py`:

    int32[256] header:
        [0] MAGIC = 2788_95051
        [1] VERSION = 1
        [2] num_tokens   (filled on close)
        [3] itemsize     (2 for uint16)
    uint16[num_tokens] tokens
    .bos.idx (sidecar): int32[] byte positions where bos_token_id occurs

Usage:
    shared/launch.sh python 01_data/tiny_shakespeare.py --out data/shakespeare
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import requests

MAGIC = 2788_95051
VERSION = 1
HEADER_SIZE = 256
BOS_ID = 50256  # GPT-2 bos == eos
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def _download(out: Path) -> str:
    if out.exists():
        return out.read_text()
    resp = requests.get(SHAKESPEARE_URL, timeout=60)
    resp.raise_for_status()
    out.write_text(resp.text)
    return resp.text


def _tokenize(text: str) -> np.ndarray:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return np.array([BOS_ID, *ids], dtype=np.uint16)


def _write_shard(out_dir: Path, name: str, tokens: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.bin"
    idx_path = out_dir / f"{name}.bos.idx"

    header = np.zeros(HEADER_SIZE, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = tokens.size
    header[3] = tokens.dtype.itemsize

    with open(bin_path, "wb") as bf:
        bf.write(header.tobytes())
        bf.write(tokens.tobytes())

    bos_positions = (
        (HEADER_SIZE * 4) + (np.where(tokens == BOS_ID)[0].astype(np.int32) * tokens.dtype.itemsize)
    )
    idx_path.write_bytes(bos_positions.tobytes())
    return bin_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/shakespeare"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    raw = args.out / "input.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading to {raw}")
    text = _download(raw)
    print(f"[data] {len(text):,} chars")

    tokens = _tokenize(text)
    print(f"[data] {tokens.size:,} tokens (uint16)")

    split = int(tokens.size * (1 - args.val_frac))
    train = tokens[:split]
    val = tokens[split:]
    train_path = _write_shard(args.out, "train", train)
    val_path = _write_shard(args.out, "val", val)

    print(f"[data] wrote {train_path} ({train.size:,} tokens)")
    print(f"[data] wrote {val_path}   ({val.size:,} tokens)")
    print(f"[data] matches {args.out}/*.bin for NanogptDataset.file_pattern")


if __name__ == "__main__":
    main()
