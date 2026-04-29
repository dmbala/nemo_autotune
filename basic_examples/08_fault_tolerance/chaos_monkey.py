"""Chaos monkey for the elastic-restart demo.

Wraps an arbitrary Python command and kills the calling process on a chosen
rank after a chosen number of seconds. Torch Elastic (torchrun) should detect
the crashed worker and respawn the whole group from the last DCP checkpoint,
up to --max-restarts times.

This is *not* a recipe patch — it's a small wrapper you launch via torchrun
instead of the training script, so the wrapper inherits ``RANK``, ``WORLD_SIZE``,
etc. from torchrun's env and then execs the real training entry.

Usage (inside the torchrun-elastic sbatch):
    python chaos_monkey.py \
        --crash-rank 1 --crash-after 30 -- \
        python /opt/Automodel/nemo_automodel/recipes/llm/train_ft.py --config <yaml>
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading


def _self_destruct_later(delay: int) -> None:
    def _kill():
        print(f"[chaos] rank {os.environ.get('RANK', '?')} self-destructing after {delay}s", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    t = threading.Timer(delay, _kill)
    t.daemon = True
    t.start()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crash-rank", type=int, default=0,
                    help="Rank that will crash. -1 = all ranks never crash (pass-through).")
    ap.add_argument("--crash-after", type=int, default=30, help="Seconds before SIGKILL.")
    ap.add_argument("child", nargs=argparse.REMAINDER,
                    help="Command to exec (prefix with --).")
    args = ap.parse_args()
    if args.child and args.child[0] == "--":
        args.child = args.child[1:]
    if not args.child:
        print("chaos_monkey: missing child command", file=sys.stderr)
        return 2

    my_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"[chaos] rank={my_rank} world={world_size} crash_rank={args.crash_rank}", flush=True)

    if args.crash_rank >= 0 and my_rank == args.crash_rank:
        _self_destruct_later(args.crash_after)

    return subprocess.call(args.child, env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main())
