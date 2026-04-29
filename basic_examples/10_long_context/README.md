# Module 10 — Long context: sequence parallel + context parallel

Training at long sequence lengths is attention-memory-bound — activations are `O(seq_len²)` for vanilla attention and `O(seq_len)` with flash. Two parallelism knobs unlock longer contexts without proportionally more GPUs:

| Technique | YAML key | What it shards | Requires |
|---|---|---|---|
| Sequence parallel (SP) | `distributed.sequence_parallel: true` | LayerNorm / dropout activations along the seq dim within a TP group | `tp_size > 1` + a TP-aware model |
| Context parallel (CP) | `distributed.cp_size: N` | The attention sequence itself, via ring-attention-style all-reduce | model supports CP; Automodel plumbs this via `components/distributed/cp_utils.py::create_context_parallel_ctx` |

Both can combine with FSDP2 / DP.

## Two demo configs

| Config | Layout | seq_len | Use |
|---|---|---|---|
| `long_ctx_sp.yaml` | dp=2, tp=2, SP on | 8,192 | Activations fit after SP sharding; shows the TP+SP combo |
| `long_ctx_cp.yaml` | dp=2, cp=2 | 16,384 | Pushes past single-GPU memory; shows the CP ring |

Both use the repo's `mock.build_unpacked_dataset` so you don't need pretokenized data — the point is the memory/throughput characteristic.

## Run

```bash
MODE=sp sbatch 10_long_context/long_ctx.slrm   # SP variant
MODE=cp sbatch 10_long_context/long_ctx.slrm   # CP variant
```

Both allocate `--nodes=1 --ntasks-per-node=4 --gres=gpu:4`.

## What to look for

**SP log** (`logs/ws10_longctx.<jobid>.out` with `MODE=sp`):
```
distributed: fsdp2 | dp=2 tp=2 cp=1 | sequence_parallel=True
[rank 0] step 0 ... tps=... mem_GiB=...
```
Compare memory use with and without `sequence_parallel: true` at the same `seq_len` to see the activation savings.

**CP log**:
```
distributed: fsdp2 | dp=2 tp=1 cp=2
[rank 0] step 0 ... seq_len=16384 ...
```
CP is the path that lets you go to seq_len > 32k on H100 without TP. A 16k run on 2×H100 uses ~25 GB/rank of activation memory.

## Combining

`dp_size × tp_size × cp_size × pp_size × ep_size × dp_replicate_size == world_size`.

Some useful combos on 4 GPUs:

| Goal | Config |
|---|---|
| Long seq, small model | `dp=1, tp=1, cp=4` |
| Long seq, medium model | `dp=2, tp=1, cp=2` (ours) |
| Wide model, medium seq | `dp=1, tp=4, cp=1, sequence_parallel=true` |
| Wide model + long seq | `tp=2, cp=2` (both halving memory on different axes) |

## Memory math

A rough rule of thumb for activation memory per rank under bf16:

```
activation_mem ≈ batch × seq_len × hidden × layers × 2 B × shard_factor
shard_factor   = 1 / (tp_size × cp_size × (SP ? tp_size : 1))  (loosely)
```

Example, Qwen3-0.6B (`hidden=1024, layers=28`), batch=1, seq=16384:
- No parallelism: ~1.8 GB activations.
- cp=2: ~0.9 GB.
- tp=2 + SP: ~0.45 GB.

Flash attention already collapses the `seq_len²` attention scores; what SP/CP save here is the *ffn activations* and the *residual / LN activations*.

## Gotchas

- **SP requires `tp_size > 1`.** It shards inside the TP group. Setting `sequence_parallel: true` with `tp_size: 1` is a no-op (and currently warns).
- **CP requires the model's attention to support it.** Qwen3 does. Custom attention modules may not — check `components/distributed/cp_utils.py` for the monkeypatch layer.
- **`dp_size: 2, tp_size: 2` on Qwen3** assumes the model's `tp_plan` is defined for its layers. If you hit "no TP plan registered for layer X", fall back to `tp_size: 1` + pure DP.
- The mock dataset yields synthetic tokens, so loss numbers are meaningless. Use this config for profiling + memory/throughput measurement (pair with Module 06's `sweep.py` to find the seq_len × batch × parallelism sweet spot). Swap in a real dataset once the memory budget is confirmed.
- **Flash attention is required for long-context in practice.** The container has it; model configs that disable it (`attn_implementation: eager`) will OOM at 16k.

## Related

- Module 03 — other distributed strategies (FSDP2/MegatronFSDP/HSDP/pipeline).
- Module 06 — sweep `(cp_size, seq_len, batch_size)` to find the throughput Pareto front.
- `/opt/Automodel/nemo_automodel/components/distributed/cp_utils.py` — CP implementation.
