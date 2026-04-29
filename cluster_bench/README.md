# cluster_bench — HPC cluster diagnostic suite built on NeMo-Automodel

Companion to the NeMo-Automodel workshop (`../workshop/`). Repurposes the workshop's benchmark recipe, sweep driver, and distributed primitives as a cluster health / acceptance / regression test suite covering:

- **Compute** — per-node MFU distribution, straggler detection
- **Network** — NCCL microbenchmarks + training-workload scaling curves
- **InfiniBand fabric** — static topology snapshots + error-counter deltas + GPU-NIC affinity
- **Storage** — DCP checkpoint throughput across filesystem tiers
- **End-to-end** — single `accept_node.slrm` that gates a node's production admission

Kempner-hardcoded (partition `kempner_dev`, storage paths under `/n/netscratch` / `/n/holylfs06`, SIF at the shared location). Container and distributed primitives come from the companion project via the `shared → ../basic_examples/shared` symlink (that project is the former "workshop/"). If it gets renamed again, re-point the symlink — nothing else changes.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Static snapshot + affinity check | ✅ built and smoke-verified on this node |
| 2 | Per-node MFU + scaling-efficiency curves | ✅ configs + sbatches built, schema-verified |
| 3 | NCCL microbenchmarks overlay + collective wrappers | ✅ overlay builder + 6-collective runner + pair matrix |
| 4 | Storage tiers + correlator + report | ✅ smoke-verified on synthetic data (scrape → correlate → report → verdicts.md) |
| 5 | MTTR + drift history (`trend.py`) | ✅ smoke-verified; regression flagging works against 7-day baseline |
| 6 | `accept_node.slrm` end-to-end gate | ✅ composed 7-stage pipeline with PASS/WARN/FAIL verdict |

## Stage 1 — static diagnostics (built)

Three read-only probes. No GPU alloc needed, runs in seconds.

```bash
# Full topology snapshot → JSON under results/snapshots/<hostname>_<ts>.json
scripts/ib_snapshot.sh

# GPU ↔ IB-HCA PCIe-NUMA affinity audit; exits non-zero if any GPU has no
# ≥NODE-quality link to an IB NIC (indicates misaffinity that costs 30-50%
# cross-node bandwidth).
scripts/affinity_check.sh

# Capture-before + capture-after IB error-counter diff around a workload:
scripts/ib_snapshot.sh results/snapshots/before.json
... run workload ...
scripts/ib_snapshot.sh results/snapshots/after.json
scripts/counter_delta.sh results/snapshots/before.json results/snapshots/after.json
# Exits non-zero if any error-class counter advanced.
```

Each snapshot captures:
- Kernel, driver, CUDA runtime versions
- Per-GPU: temp, power draw/limit, memory usage, clocks
- NVLink status (raw `nvidia-smi nvlink --status`)
- `nvidia-smi topo -m` (raw table with GPU×NIC connection qualities: NVL / PIX / PXB / PHB / NODE / SYS)
- Every IB HCA: state, phys_state, rate (e.g. `400 Gb/sec (4X NDR)`), link_layer, full `/sys/class/infiniband/.../counters/` dump
- `ibv_devinfo -v` raw output per HCA
- `ibdev2netdev` mapping

### `affinity_check.sh` quality rubric

From `nvidia-smi topo -m`, descending order (best → worst):

| Symbol | Means | Verdict |
|---|---|---|
| `NVL` / `NV<N>` | Direct NVLink | OK (won't apply to GPU↔NIC, only GPU↔GPU) |
| `PIX` | Same PCIe switch | OK (ideal for GPU↔NIC) |
| `PXB` | Multiple PCIe bridges, same root complex | OK |
| `PHB` | Via PCI host bridge | OK |
| `NODE` | Same NUMA node (via PHB) | OK (baseline acceptable) |
| `SYS` | Across UPI/QPI between sockets | **WARN** — 30-50% BW loss |
| `X` | None / unreachable | **FAIL** |

This node (holygpu8a10301, 1-GPU alloc): GPU0 best link `PXB` to NIC4 → `OK`.

### `counter_delta.sh` error classes

Non-zero delta on any of these → FAIL:

```
symbol_error, link_downed, port_rcv_errors, port_rcv_remote_physical_errors,
port_rcv_switch_relay_errors, port_xmit_discards, port_xmit_constraint_errors,
port_rcv_constraint_errors, local_link_integrity_errors,
excessive_buffer_overrun_errors, VL15_dropped, link_error_recovery
```

Benign traffic counters (`port_xmit_data`, `port_rcv_data`, `*_packets`) are printed but don't trigger FAIL.

## What's hardcoded for Kempner

- Tools expected on host: `nvidia-smi`, `ibv_devinfo`, `ibstat`, `ibdev2netdev`, Python ≥ 3.6 in `/usr/bin/python3`.
- No assumptions yet about which HCA names map to which rails — but `mlx5_0`/`mlx5_1` are Ethernet management (ignored), `mlx5_2`-`mlx5_5` are 4× NDR 400 Gb/s InfiniBand on this cluster.

## Stage 2 — training baselines (built)

Two benchmark configs + two sbatch drivers:

| Config | Model | Purpose |
|---|---|---|
| `configs/compute_baseline.yaml` | Qwen3-1.7B, 1 GPU | Per-node MFU baseline for straggler detection |
| `configs/network_scaling.yaml` | Qwen2.5-7B, world size sweep | Intra-node (NVLink) + inter-node (IB) scaling |

```bash
# One node at a time — submit for each target node:
for n in holygpu8a10301 holygpu8a10302 holygpu8a10303; do
  sbatch --nodelist=$n sbatch/per_node.slrm
done

# World-size scaling on a 2-node allocation — runs 1×1, 1×4, 2×4 in sequence:
sbatch sbatch/scaling.slrm
```

Each run produces a `benchmark_results.json` (MFU, TFLOPs/GPU, avg_iter_time_seconds) under `$RESULTS_ROOT/cluster_bench/{per_node,scaling}/<ts>/`. `per_node.slrm` wraps the benchmark with pre/post `ib_snapshot.sh` + `counter_delta.sh` so fabric hiccups during the run get caught automatically.

## Stage 3 — NCCL microbenchmarks (built)

```bash
# One-time, on a login node:
bash nccl_tests/install_overlay.sh       # writes nccl_tests/overlay.img (~2 GB), builds nccl-tests MPI=0

# Intra-node (all 6 collectives × 1 node × 4 GPU):
sbatch nccl_tests/intra_node.slrm

# Inter-node allreduce on 2 specific nodes (for heatmaps):
COLLECTIVE=all_reduce sbatch --nodelist=nodeA,nodeB nccl_tests/inter_node.slrm

# All-pairs 2-node heatmap:
bash nccl_tests/pair_matrix.sh "nodeA nodeB nodeC nodeD"    # submits N*(N-1)/2 pairs
```

Each run emits parsed CSVs + appends rows into `history.jsonl` keyed on `kind=nccl_perf`.

## Stage 4 — Storage + correlator (built)

```bash
# Per-tier DCP-save throughput (defaults: tmp + netscratch + holylfs):
sbatch sbatch/storage.slrm

# Collate all JSONs → unified history.jsonl:
shared/launch.sh python analysis/scrape_metrics.py --results $RESULTS_ROOT/cluster_bench --out results/history.jsonl

# Per-node verdict (training × NCCL × storage cross-reference):
shared/launch.sh python analysis/correlate.py --history results/history.jsonl --out results/verdicts.md

# Cluster-wide leaderboard:
shared/launch.sh python analysis/report.py --history results/history.jsonl --out results/history.md
```

The correlator flags any host where MFU is >5% below the cluster p50 *and*
attributes the cause: DEGRADED_COMPUTE / DEGRADED_NCCL / SLOW_STORAGE /
DEGRADED_FABRIC / MULTIPLE.

## Stage 5 — MTTR + drift (built)

```bash
# MTTR measurement per node:
sbatch sbatch/mttr.slrm

# Time-series trend with sparklines + regression flagging:
shared/launch.sh python analysis/trend.py --history results/history.jsonl \
    --out results/history_trend.md --regressions results/regressions.md --threshold 5.0
```

`trend.py` groups by (host, day), computes 7-day p50 baseline per host×metric,
flags any day where the latest value is below baseline by > threshold.
Sparkline column (`▁▂▃▄▅▆▇█`) shows the last 14 days at a glance.

## Stage 6 — Acceptance gate (built)

```bash
sbatch --nodelist=<candidate_node> sbatch/accept_node.slrm
```

Runs 7 stages in order: snapshot → affinity → compute MFU → NCCL all_reduce →
storage to /tmp → MTTR → counter delta. Emits `verdict.json` with `PASS` /
`WARN` / `FAIL` + per-stage breakdown. Default thresholds (override via env):

| Threshold | Default | What it gates |
|---|---|---|
| `ACCEPT_MFU_FLOOR` | 40 | Min MFU% on Qwen3-1.7B single-GPU baseline |
| `ACCEPT_NCCL_BUSBW_FLOOR` | 150 GB/s | Min all_reduce busbw on 4 GPUs at 1 GB msg |
| `ACCEPT_MTTR_CEILING` | 120 s | Max wall-clock for kill + resume |

Also appends an `accept_node` row to `history.jsonl` so every admission test
becomes part of the cluster's drift history automatically.

## Results layout

```
results/
  snapshots/<host>_<ts>.json     # static topology dumps (Stage 1)
  history.jsonl                  # append-only time series (Stages 2+)
  history.md                     # generated trend report
  regressions.md                 # flagged regressions > 5%
  raw/<jobid>/                   # per-job artifacts (nccl CSVs, nsys reps)
```
