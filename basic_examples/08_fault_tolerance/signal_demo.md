# Signal-safe saves and why Automodel's recipe doesn't do them

On shared HPC clusters, the practical ways a training job dies are:

1. **SLURM preemption** — Slurm sends `SIGTERM` (then `SIGKILL` after a grace period) when your job is preempted by a higher-priority job.
2. **Wall-time exhaustion** — also `SIGTERM` followed by `SIGKILL` at `--time` expiry.
3. **Node failure** — hardware error; the job just disappears. No signal.
4. **NCCL hang** — one rank stalled; the collective timeout (`dist_env.timeout_minutes`) trips and raises.
5. **OOM** — process dies. No save.

In all of these, the **only reliable recovery mechanism is the last on-disk checkpoint**. That's why Modules 02/03 use `save_consolidated: true` and why this module uses `ckpt_every_steps: 10`.

## The "save-on-signal" pattern

Ideal: the training loop installs a handler for `SIGTERM`, writes one final checkpoint on receipt, and exits cleanly. You get the checkpoint at step N-epsilon instead of N-K (where K is the checkpoint interval).

Automodel actually ships a utility for this: **`nemo_automodel.components.training.signal_handler.DistributedSignalHandler`** (`/opt/Automodel/nemo_automodel/components/training/signal_handler.py:90`). It's a context manager that installs a handler, coordinates the flag across ranks via `all_gather`, and restores the original handler on exit.

```python
from nemo_automodel.components.training.signal_handler import DistributedSignalHandler

with DistributedSignalHandler(signal.SIGTERM) as h:
    for step in range(max_steps):
        loss = train_step(...)
        if any(h.signals_received()):
            self.save_checkpoint(step)
            return
```

**But the recipe (`recipes/llm/train_ft.py`) does not use it.** `grep -rn signal_handler recipes/` returns nothing. So when you land in a real SLURM-preemption situation, expect to resume from whatever the last `ckpt_every_steps` saved — not from the exact step at termination.

## Practical advice

- **Tune `ckpt_every_steps`**: frequent enough that losing one interval's progress is tolerable, infrequent enough that the checkpoint I/O doesn't dominate. With async DCP (`is_async: true`, Module 03), you can check-point 5–10× more often for the same overhead.
- **Rely on `restore_from: LATEST`** to auto-resume across restarts. This module exercises that path.
- **Wrap sbatch scripts** with `#SBATCH --signal=B:USR1@120` if you want Slurm to send `SIGUSR1` two minutes before wall-time cutoff, giving your job time to checkpoint. The recipe would still need to handle `SIGUSR1` — see next bullet.
- **If you need true save-on-preemption**, patch the recipe: import `DistributedSignalHandler`, wrap `run_train_validation_loop`'s inner loop with the context manager, and call `self.save_checkpoint(...)` when a signal is seen. ~15 lines. Worth contributing upstream.

## NCCL timeouts

`dist_env.timeout_minutes` (passed through to `init_process_group(timeout=timedelta(minutes=N))`) controls how long a rank will wait on a collective before raising. Default in Automodel examples is `1` (minute), which is aggressive. The `tiny_resume.yaml` in this module uses `10` so transient stalls don't fail a job that is otherwise healthy.

If you see `NCCL WARN Caught collective operation failure due to timeout`, raise `timeout_minutes`. Don't disable the timeout — hung ranks should error, not hang forever.
