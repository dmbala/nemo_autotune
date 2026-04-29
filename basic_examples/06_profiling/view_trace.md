# Viewing profiler traces

## nsys traces

`run_nsys.slrm` produces `$RESULTS_ROOT/nsys/trace.nsys-rep` on the cluster.

### Option 1 — summarize on the cluster

```bash
# Inside the container (has nsys CLI):
shared/launch.sh nsys stats $RESULTS_ROOT/nsys/trace.nsys-rep | head -40
shared/launch.sh nsys stats --report gpukernsum $RESULTS_ROOT/nsys/trace.nsys-rep
```

Common reports:
- `gpukernsum` — kernel-level time breakdown.
- `gputrace` — detailed CUDA API calls.
- `nvtxsum` — time per NVTX range (maps to Automodel's per-step ranges).

### Option 2 — open in NVIDIA Nsight Systems GUI

Copy the `.nsys-rep` to your laptop and open with the desktop Nsight Systems UI:

```bash
scp cluster:$RESULTS_ROOT/nsys/trace.nsys-rep ./
# on your laptop:
nsys-ui trace.nsys-rep
```

The timeline shows each rank's CPU/GPU lanes, CUDA kernels, NVTX ranges (look for `iteration`, `optimizer`, `forward_backward_*`), and NCCL traffic.

## torch.profiler traces

`torch_profiler_patch.py` writes a directory of `events.json.gz` chrome-trace files under `$RESULTS_ROOT/torch_profiler/`.

### Option 1 — chrome://tracing

```bash
scp cluster:$RESULTS_ROOT/torch_profiler/<host>_<pid>_*.json.gz ./
```
1. Open Chrome and go to `about://tracing` (or `chrome://tracing`).
2. Click **Load** and pick the `.json.gz`.
3. Use `w`/`s` to zoom, `a`/`d` to pan, `?` for help.

### Option 2 — TensorBoard Profiler (if the plugin is installed)

```bash
pip install torch_tb_profiler    # inside a user venv, outside the container is easiest
tensorboard --logdir $RESULTS_ROOT/torch_profiler
```
Open http://localhost:6006/#pytorch_profiler.

## What to look for

- **Bubbles** between forward and backward on small models → kernel-launch overhead; raise `local_batch_size` or enable FSDP2 overlap.
- **All-gather / reduce-scatter** dominating step time → communication-bound; try `activation_checkpointing: true` or larger `dp_size`.
- **Optimizer time ≳ forward+backward** → your model is tiny relative to the optimizer state; consider fused AdamW.
- **NVTX range `forward_backward_0` much slower than `_1`..`_N`** → warmup / JIT; use `benchmark.warmup_steps >= 5`.
