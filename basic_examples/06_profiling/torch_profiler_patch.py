"""Standalone torch.profiler overlay.

Automodel's benchmark recipe only integrates with nsys (NVTX + cudaProfilerStart/Stop).
When you want the torch.profiler Chrome-trace view — for Python-side stack frames,
op-level breakdown, and TensorBoard Profiler support — run this instead.

This script is intentionally *not* patched into the recipe. It's a self-contained
forward/backward loop that loads a model the same way the recipe does, wraps a
configurable window of steps with torch.profiler, and writes a trace under
$RESULTS_ROOT/torch_profiler/.

Usage:
    shared/launch.sh python 06_profiling/torch_profiler_patch.py \
        --model Qwen/Qwen3-0.6B --seq-len 1024 --batch-size 4 --steps 12 \
        --out $RESULTS_ROOT/torch_profiler
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from transformers import AutoConfig, AutoModelForCausalLM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=12, help="Total steps including profile window")
    ap.add_argument("--wait", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--active", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("./torch_profiler_trace"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_config(cfg).to(device=device, dtype=torch.bfloat16)
    model.train()

    vocab = cfg.vocab_size
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    sched = schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=1)
    loss_fn = torch.nn.CrossEntropyLoss()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=tensorboard_trace_handler(str(args.out)),
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        for step in range(args.steps):
            ids = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)
            labels = ids.clone()

            out = model(input_ids=ids)
            logits = out.logits if hasattr(out, "logits") else out
            loss = loss_fn(logits.view(-1, vocab).float(), labels.view(-1))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            prof.step()
            print(f"[prof] step {step}: loss={loss.item():.4f}")

    print(f"[prof] trace written to {args.out}")
    print(f"[prof] open in Chrome: about://tracing  →  load  →  pick trace json.gz")
    print(f"[prof] or TensorBoard: tensorboard --logdir {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
