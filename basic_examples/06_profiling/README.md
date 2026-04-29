# Module 06 — Performance & profiling (+ autotuner grid sweep)

Three pieces:

1. **`nsys` via Automodel's benchmark recipe** — the only profiling integration Automodel ships. NVTX ranges + `cudaProfilerStart/Stop` around a configurable step window.
2. **`torch.profiler` overlay** — Automodel doesn't wire `torch.profiler`, so this module provides a standalone `torch_profiler_patch.py` that loads the same model shape and wraps N steps in a profile.
3. **Autotuner grid sweep** — Automodel has no autoconfigurator. The benchmark recipe emits `benchmark_results.json` (MFU%, TFLOPs/GPU, iter time); `sweep.py` drives it over a grid and `report.py` tabulates the winners.

## Nsys profiling

```bash
sbatch 06_profiling/run_nsys.slrm
```

The `configs/nsys_gpt2_124m.yaml` sets:

```yaml
benchmark:
  nsys_start: 8
  nsys_end: 12
  nsys_ranks: [0]
```

The recipe calls `torch.cuda.cudart().cudaProfilerStart()` at step 8 on rank 0 and `cudaProfilerStop()` at step 12. Our sbatch wraps the whole run with `nsys profile -c cudaProfilerApi` so only those five steps end up in the trace.

Output: `$RESULTS_ROOT/nsys/trace.nsys-rep`. Open with Nsight Systems GUI or `nsys stats` — see `view_trace.md`.

## torch.profiler overlay

```bash
shared/launch.sh python 06_profiling/torch_profiler_patch.py \
    --model Qwen/Qwen3-0.6B --seq-len 1024 --batch-size 4 \
    --steps 12 --wait 2 --warmup 3 --active 5 \
    --out $RESULTS_ROOT/torch_profiler
```

Writes Chrome-trace JSONs for TensorBoard or `chrome://tracing`. Schedule is `skip(wait) → warmup(warmup) → record(active)` repeated once. Use `with_stack=True` if you want Python-level frames (slower).

This script is *deliberately not* a patch to `train_ft.py` — it's a sidecar that uses the same HF config. That keeps the recipe untouched and the profiler code visible.

## Autotuner grid sweep

```bash
sbatch 06_profiling/sweep.slrm          # 8 points × ~1 min each
# or directly:
shared/launch.sh python 06_profiling/sweep.py \
    --base 06_profiling/configs/benchmark_base.yaml \
    --results $RESULTS_ROOT/sweep \
    --batch-sizes 1,2,4,8 --seq-lens 512,1024 --dp-tp 4x1 \
    --nproc-per-node 4 --max-steps 20
```

For each grid point the sweep:
1. Deep-copies `benchmark_base.yaml`, applies the overrides (`step_scheduler.local_batch_size`, `dataset.seq_len`, `distributed.dp_size/tp_size`), and sets `benchmark.json_output_path` to a per-run file.
2. Launches `automodel -c <tmp> benchmark llm --nproc-per-node=N` as a subprocess.
3. Parses the per-run JSON (the recipe writes `avg_iter_time_seconds`, `avg_mfu_percent`, `tflops_per_gpu`).

After the sweep, `report.py` ranks runs by MFU% and writes `summary.md`:

```
| rank | rid       | local_bs | seq_len | dp×tp | MFU%  | TFLOPs/GPU | iter_s |
|------|-----------|----------|---------|-------|-------|------------|--------|
| 1    | a1b2c3d4  | 8        | 1024    | 4x1   | 42.13 | 416.67     | 0.6421 |
| 2    | 5e6f7a8b  | 4        | 1024    | 4x1   | 39.88 | 394.40     | 0.3389 |
...
```

### Extending the grid

`sweep.py` enforces `dp_size × tp_size == --nproc-per-node`, so:

- For world_size=4: `--dp-tp 4x1,2x2,1x4`
- For world_size=8 (multi-node): bump to `--dp-tp 8x1,4x2,2x4` and use `--nodes=2 --ntasks-per-node=4`.

For bigger sweeps, wrap each grid row in a Slurm array job: emit the overrides once, then `sbatch --array=0-N sweep_one.slrm`.

## Gotchas

- `benchmark.peak_tflops` must match your hardware. H100/H200 = 989 (BF16 dense). A100 = 312. MFU is computed relative to this.
- The benchmark recipe injects `vocab_size` into `cfg.dataset` by loading `AutoConfig.from_pretrained(cfg.model.config.pretrained_model_name_or_path)`. For a custom random-init model without a Hub id, this path doesn't run — use a real HF model in the benchmark config.
- `nsys` overhead is low, but enabling it during warmup skews the first couple of timed iterations. Keep `nsys_start >= warmup_steps + 2`.
- `sweep.py` launches runs sequentially in a single Slurm allocation. If a run hangs, the whole sweep hangs. Add `--max-steps` ≤ 30 to bound wall time.
