# Module 00 — Setup & Smoke Test

Confirms the container works on this cluster and surfaces the knobs the rest of the workshop will use.

## Run it

```bash
# One-shot on a GPU node:
sbatch 00_setup/smoke_test.slrm

# Or interactively (needs a GPU allocation):
salloc -p kempner_dev --gres=gpu:1 --mem=32G --time=00:10:00
shared/launch.sh python 00_setup/smoke_test.py
```

## What you should see

```
python: 3.12.3
[ OK ] nemo_automodel: 0.3.0
[ OK ] torch: 2.10.0a0+b558c986e8.nv25.11
[ OK ] transformers: 5.0.0
...
cuda available: True  devices: 1
device 0: NVIDIA H200
bf16 matmul ok, |y|=...
FSDP2Config fields: ['sequence_parallel', 'tp_plan', 'defer_fsdp_grad_sync', ...]
CLI COMMAND_ALIASES: {'finetune': 'train_ft', 'pretrain': 'train_ft', 'benchmark': 'benchmark'}
```

If you see `Permission denied: '/opt/Automodel/nemo_automodel/__init__.py'`, you're using the stock NGC image instead of the `-fixed` variant. Fix: set `SIF=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/nemo/nemo-automodel-26.02-fixed.sif` before launching, or add `--fakeroot` to `singularity exec`.

## Key takeaways

- `automodel` is a thin launcher for `nemo_automodel/recipes/<domain>/<recipe>.py`. `finetune` and `pretrain` both route to `train_ft.py`. The config decides behavior.
- The container's venv is at `/opt/venv`. You must `source /opt/venv/env.sh` before running `python` or `automodel`. The `shared/launch.sh` wrapper does that for you.
- `torch.distributed` is available and bf16 matmul works on H100/H200. Modules 02/03/04/06 will exercise multi-GPU paths.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `automodel: command not found` inside container | Venv not sourced. Use `shared/launch.sh`, don't call `singularity exec` yourself. |
| `bitsandbytes: FAIL` | Not a blocker for this workshop; `bitsandbytes` is only needed for 8-bit optimizers. |
| Job fails with `Unable to open PTX` / CUDA arch mismatch | This container is built for Hopper. If you land on an older partition, switch to H100/H200. |
