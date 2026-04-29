# Module 08 — Fault tolerance & distributed checkpoint resume

Extends Module 03's distributed checkpointing story with three concrete fault-tolerance demos and a doc on signal-safe saves.

## What Automodel gives you

| Primitive | Where |
|---|---|
| `checkpoint.restore_from: LATEST` → auto-resume from last saved step | `/opt/Automodel/nemo_automodel/recipes/base_recipe.py:143` |
| `LATEST` symlink kept current after every save | `base_recipe.py:410` |
| DCP reshardable loads (save at dp=N, load at dp=M) | `components/checkpoint/checkpointing.py::dcp.load` |
| Async DCP save (torch ≥ 2.9) — non-blocking on hot path | `CheckpointingConfig.is_async` |
| `dist_env.timeout_minutes` — NCCL timeout for stuck collectives | `components/distributed/init_utils.py:126` |
| `DistributedSignalHandler` (utility, **not wired into the recipe**) | `components/training/signal_handler.py` |

## What you'll exercise

| Demo | sbatch | What it shows |
|---|---|---|
| **Kill & resume** | `kill_and_resume.slrm` | Train tiny GPT-2 until the first checkpoint lands, SIGKILL the whole process group, relaunch with same config → `restore_from: LATEST` picks up from the last `ckpt_every_steps` boundary. Smoke-verified end-to-end. |
| **DCP reshard** | `reshard_demo.slrm` | Save at `dp_size=4` on 4 GPUs. Resume same checkpoint at `dp_size=2` on 2 GPUs. DCP stitches shards on load and re-shards to the new mesh. |
| **Torch elastic** | `elastic_restart.slrm` | Launch with `torchrun --max-restarts=3`; `chaos_monkey.py` kills rank 1 after 45 s. Elastic respawns the group, recipe's `restore_from: LATEST` restores state, training continues. |

## Requires

- Module 01 tiny data (`data/shakespeare/train.bin`). No HF downloads needed.

## Run

```bash
# 1-GPU kill-and-resume (fastest, most instructive):
sbatch 08_fault_tolerance/kill_and_resume.slrm

# 4-GPU reshardable load:
sbatch 08_fault_tolerance/reshard_demo.slrm

# 4-GPU elastic restart with injected crash:
sbatch 08_fault_tolerance/elastic_restart.slrm
```

You can also run `kill_and_resume.sh` interactively on an existing GPU allocation:

```bash
salloc -p kempner_dev --gres=gpu:1 --mem=64G --time=00:30:00
./08_fault_tolerance/kill_and_resume.sh \
    $SCRATCH_ROOT/data/shakespeare/train.bin \
    $SCRATCH_ROOT/checkpoints/ft_kill_resume \
    25
```

## What to look for in the logs

**Kill & resume** (`logs/ws08_kill_resume.*.out`) — actual output from the dev node:

```
=== PHASE 1: launching training, SIGKILL after first checkpoint ===
[chaos] runner PID=3807787 PGID=3807787
[chaos] waiting up to 180s for first checkpoint ...
[chaos] first checkpoint: epoch_0_step_99 (after 46s)
[chaos] ckpts at time of kill: .../epoch_0_step_299
[chaos] sending SIGKILL to process group 3807787

=== PHASE 2: relaunching — expecting restore_from: LATEST ===
[resume] LATEST -> epoch_0_step_299
Loading checkpoint from .../epoch_0_step_299
step 300 | epoch 0 | loss 6.5438 | ...   ← continues at step 300, loss consistent with resume
```

Fresh-init loss at step 0 would be ~10.80; seeing 6.54 at "step 300" confirms the optimizer + model state were restored, not re-initialized.

**DCP reshard** (`logs/ws08_reshard.*.out`):

```
# Phase 1, 4 ranks
Rank 0/4: saving shard-00001-model-00001-of-00001.safetensors
Rank 1/4: saving shard-00002-...
...
# Phase 2, 2 ranks
Rank 0/2: loading shard-00001-model-00001-of-00001.safetensors
Rank 1/2: loading shard-00002-...
step 20 | ...  ← resumed at step 20 on new mesh
```

**Elastic** (`logs/ws08_elastic.*.out`):

```
[chaos] rank=1 world=4 crash_rank=1
[rank 1] step 8 | loss 10.14 ...
[chaos] rank 1 self-destructing after 45s
torchrun: [ERROR] worker pid 5678 exited with code SIGKILL, restarting group
[rank 0] Loading checkpoint from .../epoch_0_step_9
[rank 0] step 10 | loss 10.28 ...  ← group respawned, resumed from ckpt
```

## Gotchas

- **Kill timing**: `kill_and_resume.sh` waits for the first checkpoint to land rather than using a fixed timer. That avoids killing during container spin-up / Python imports (which can take 15–30 s). If the first checkpoint takes longer than `MAX_WAIT` (180 s default), the script gives up. Tune `ckpt_every_steps` down for faster demos.
- **setsid + kill `-$PGID`**: `SIGKILL` to the shell's direct child won't propagate through `singularity exec → torchrun → python`. The script wraps phase 1 in `setsid` and kills the whole process group.
- `restore_from: LATEST` requires `ckpt_every_steps` to have fired at least once; otherwise there's nothing to resume from and the recipe starts fresh (that's the *right* behavior, just be aware).
- **DCP reshard only works for the model/optimizer shards** — the dataloader state is pickled per rank. Resuming at a different world size resets the data cursor. For a rigorous demo, set `dataloader.shuffle: false` so the starting position is deterministic.
- `chaos_monkey.py` uses `SIGKILL` (simulates a hard crash / preemption via `scancel --signal=KILL`). For `SIGTERM` (clean preemption), you'd want `DistributedSignalHandler` wired into the recipe — see `signal_demo.md` for the how and why.
- Torch elastic respawns within a **single Slurm allocation**. If the whole allocation dies (wall-time, node failure), Slurm's own requeue (`--requeue` + `sbatch --requeue`) plus `restore_from: LATEST` is the recovery path.
- `--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT` on a single-node job points at the local host. For multi-node elastic, the rendezvous endpoint must be the job's first node and all nodes must agree (use `scontrol show hostnames | head -1`). Covered in Module 03's `run_hsdp_multinode.slrm`.

## Related reading

- `signal_demo.md` — why the recipe doesn't save on `SIGTERM`, what you'd patch to make it.
- Module 03 `README.md` — base DCP save/load format, async save, HF consolidation.
- PyTorch Elastic docs: https://pytorch.org/docs/stable/elastic/run.html
