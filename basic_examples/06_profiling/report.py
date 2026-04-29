"""Rank grid-sweep results from sweep.py and write a markdown leaderboard.

Usage:
    shared/launch.sh python 06_profiling/report.py --results $RESULTS_ROOT/sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output markdown path (default: <results>/summary.md)")
    args = ap.parse_args()

    index_path = args.results / "index.json"
    if not index_path.exists():
        raise SystemExit(f"no index.json under {args.results}; run sweep.py first")
    runs = json.loads(index_path.read_text())
    if not runs:
        raise SystemExit("no successful runs")

    runs.sort(key=lambda r: -float(r.get("avg_mfu_percent", 0) or 0))

    rows = []
    header = ["rank", "rid", "local_bs", "seq_len", "dp×tp", "MFU%", "TFLOPs/GPU", "iter_s"]
    rows.append("| " + " | ".join(header) + " |")
    rows.append("|" + "|".join("---" for _ in header) + "|")
    for i, r in enumerate(runs, 1):
        ov = r.get("_overrides", {})
        dp = ov.get("distributed.dp_size") or "auto"
        tp = ov.get("distributed.tp_size", 1)
        rows.append("| " + " | ".join([
            str(i),
            r.get("_rid", "?"),
            str(ov.get("step_scheduler.local_batch_size", "?")),
            str(ov.get("dataset.seq_len", "?")),
            f"{dp}x{tp}",
            _fmt(r.get("avg_mfu_percent")),
            _fmt(r.get("tflops_per_gpu")),
            _fmt(r.get("avg_iter_time_seconds"), 4),
        ]) + " |")

    md = f"# Sweep leaderboard ({len(runs)} runs)\n\nSorted by MFU% (descending).\n\n" + "\n".join(rows) + "\n"
    out = args.out or (args.results / "summary.md")
    out.write_text(md)
    print(f"[report] wrote {out}")
    print(md)


if __name__ == "__main__":
    main()
