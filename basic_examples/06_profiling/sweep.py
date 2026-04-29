"""Autotuning via grid sweep over training configs, driven by Automodel's
BenchmarkingRecipeForNextTokenPrediction.

Automodel has no built-in autoconfigurator. Its benchmark recipe is the
measurement primitive; this script is the search loop.

For each point in the grid we:
  1. Write a temp YAML that sets ``--benchmark.json_output_path`` to a per-run file.
  2. Launch the benchmark via ``shared/launch.sh automodel -c <tmp> benchmark llm``.
  3. Collect ``avg_iter_time_seconds``, ``avg_mfu_percent``, ``tflops_per_gpu`` from the JSON.

After sweeping, call ``report.py`` on the results dir to get a sorted markdown table.

Usage:
    shared/launch.sh python 06_profiling/sweep.py \
        --base 06_profiling/configs/benchmark_base.yaml \
        --results $RESULTS_ROOT/sweep \
        --batch-sizes 1,2,4,8 --seq-lens 512,1024 --dp-tp "4x1,2x2,1x4"

Notes:
  - ``--dp-tp`` is a comma-separated list of ``dp_size x tp_size`` pairs;
    their product must equal your world size (set via ``--nproc-per-node``).
  - Runs are sequential (single SLURM allocation). For a bigger sweep, wrap this
    script in a job array.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _parse_dp_tp(s: str) -> list[tuple[int, int]]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        dp, tp = tok.lower().split("x")
        out.append((int(dp), int(tp)))
    return out


def _update_in_place(cfg: Any, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {}) if isinstance(d, dict) else getattr(d, k)
    if isinstance(d, dict):
        d[keys[-1]] = value
    else:
        setattr(d, keys[-1], value)


def _make_run_yaml(base: dict, out_path: Path, overrides: dict) -> Path:
    cfg = copy.deepcopy(base)
    for k, v in overrides.items():
        _update_in_place(cfg, k, v)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


def _run_id(overrides: dict) -> str:
    key = json.dumps(overrides, sort_keys=True).encode()
    return hashlib.sha1(key).hexdigest()[:10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--batch-sizes", default="1,2,4",
                    help="Comma list of local_batch_size values")
    ap.add_argument("--seq-lens", default="512,1024", help="Comma list of seq_len values")
    ap.add_argument("--dp-tp", default="4x1,2x2,1x4",
                    help="Comma list of <dp_size>x<tp_size>. Product = world_size.")
    ap.add_argument("--nproc-per-node", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=30,
                    help="Per-run max_steps (keep small for quick sweeps)")
    ap.add_argument("--warmup-steps", type=int, default=5)
    args = ap.parse_args()

    base = yaml.safe_load(args.base.read_text())
    args.results.mkdir(parents=True, exist_ok=True)

    workshop_root = Path(__file__).resolve().parents[1]
    launch = workshop_root / "shared" / "launch.sh"

    grid = []
    for bs in map(int, args.batch_sizes.split(",")):
        for sl in map(int, args.seq_lens.split(",")):
            for dp, tp in _parse_dp_tp(args.dp_tp):
                if dp * tp != args.nproc_per_node:
                    print(f"[sweep] skip dp={dp} tp={tp} (product != {args.nproc_per_node})")
                    continue
                point = {
                    "step_scheduler.local_batch_size": bs,
                    "step_scheduler.global_batch_size": bs * dp,
                    "step_scheduler.max_steps": args.max_steps,
                    "benchmark.warmup_steps": args.warmup_steps,
                    "dataset.seq_len": sl,
                    "distributed.tp_size": tp,
                }
                # With tp > 1 we let dp_size auto-resolve from world_size so we
                # don't emit a null into the YAML.
                if tp == 1:
                    point["distributed.dp_size"] = dp
                grid.append(point)

    if not grid:
        print("[sweep] empty grid", file=sys.stderr)
        return 2

    completed = []
    for i, overrides in enumerate(grid, 1):
        rid = _run_id(overrides)
        rdir = args.results / rid
        rdir.mkdir(parents=True, exist_ok=True)
        json_path = rdir / "benchmark_results.json"
        cfg_copy = dict(overrides)
        cfg_copy["benchmark.json_output_path"] = str(json_path)
        cfg_yaml = _make_run_yaml(base, rdir / "config.yaml", cfg_copy)

        print(f"\n[sweep] ({i}/{len(grid)}) rid={rid} overrides={overrides}")
        cmd = [str(launch), "automodel", "-c", str(cfg_yaml), "benchmark", "llm",
               f"--nproc-per-node={args.nproc_per_node}"]
        with (rdir / "stdout.log").open("w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)

        if rc != 0 or not json_path.exists():
            print(f"[sweep] run {rid} failed (rc={rc})")
            (rdir / "status.txt").write_text(f"failed rc={rc}\n")
            continue
        with json_path.open() as f:
            summary = json.load(f)
        summary.update({"_overrides": overrides, "_rid": rid})
        completed.append(summary)
        (rdir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[sweep] {rid}: MFU={summary.get('avg_mfu_percent', '?')}  "
              f"TFLOPs/GPU={summary.get('tflops_per_gpu', '?')}")

    index = args.results / "index.json"
    index.write_text(json.dumps(completed, indent=2))
    print(f"\n[sweep] done. {len(completed)}/{len(grid)} runs succeeded → {index}")
    print(f"[sweep] run  shared/launch.sh python 06_profiling/report.py --results {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
