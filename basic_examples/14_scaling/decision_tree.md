# Scaling decision tree — which knob, when?

Rough mental model: memory is the dominant constraint, not compute. As the model grows, three memory buckets grow at different rates:

| Bucket | Grows with | Sharded by |
|---|---|---|
| **Parameters** | model size | FSDP2 (`dp_size`), TP (`tp_size`), PP (`pp_size`) |
| **Optimizer state** | model size × 4–16× | FSDP2 (automatically) |
| **Activations** | model size × seq_len × batch | Activation ckpt, SP, CP |

Apply knobs in roughly this order of "least disruptive first":

```
Is the model <1B?
├─ yes → single GPU, no sharding. (Module 02 tiny config.)
│
Does it fit on one GPU after FSDP2 dp=4?
├─ yes  → Step 2. (~1.7B)
├─ no (optimizer state OOM) → + activation_checkpointing.    Step 3. (~7B)
│
Does activations still OOM after AC?
├─ yes → + TP and SP.                                        Step 4. (~32B)
│
Is the model a MoE?
├─ yes → + EP (replace FSDP2 expert replication).            Step 5. (MoE 30B)
│
Do weights still not fit after dp + tp + ac?
├─ yes → + PP across nodes.                                  Step 6. (70B dense)
│
Is the model multi-hundred B?
└─ yes → full stack: FSDP + TP + EP + PP, multi-node, TE/DeepEP. Step 7. (120B MoE)
```

## Knob-by-knob cheat sheet

**FSDP2 (`distributed.dp_size`)** — always on for >1 GPU. Shards parameters, grads, and optimizer state across DP ranks. Cost: 2 all-gathers + 1 reduce-scatter per step.

**Activation checkpointing (`distributed.activation_checkpointing`)** — set `true` when activations OOM. Recomputes activations during backward. Cost: ~30 % extra forward FLOPs. Turn on ≥ 7B dense / ≥ 2k seq_len.

**Tensor parallel (`distributed.tp_size`)** — shards weight matrices (qkv, ffn) across TP ranks. Needed when a single weight matrix is too big for one GPU. Requires the model to have a `tp_plan`; most HF models work, custom ones need one defined. Cost: all-reduce inside every transformer block.

**Sequence parallel (`distributed.sequence_parallel`)** — shards LayerNorm / dropout activations across TP ranks. *Only enable with tp > 1*, where it's essentially free memory savings.

**Expert parallel (`distributed.ep_size`)** — for MoE only. Shards experts across EP ranks (one or a handful per rank) rather than replicating them on every DP rank. Without EP, a 30B MoE with 128 experts replicates 30 GB of experts on every DP rank.

**Pipeline parallel (`distributed.pp_size`)** — shards *layers* across PP ranks. Fundamentally different from TP — PP runs depth-wise, TP runs width-wise. Use PP when your model depth × weight size exceeds what a single node can hold even after TP. Cost: pipeline bubbles (schedulable by `pp_schedule: interleaved1f1b`).

**Context parallel (`distributed.cp_size`)** — shards the sequence dimension during attention. Use for seq_len > 16k. See Module 10.

## Arithmetic rules that save you trouble

`dp_size × dp_replicate_size × tp_size × pp_size × cp_size × ep_size == world_size`.

For a given world size, pick knobs such that the product equals it. Setting `dp_size: none` (auto) lets Automodel fill in the remainder.

## "I got an OOM" — what to flip

| Phase of OOM | Try |
|---|---|
| During `init_model` on rank 0 | The full model didn't fit before sharding. Upgrade to FSDP2 → + TP → + PP. |
| During the first forward | Activations. Enable `activation_checkpointing: true`, or reduce `local_batch_size` / `seq_len`. |
| During the first backward | Grads + optimizer state. Raise `dp_size`. |
| Inconsistent per-rank OOM on MoE | Expert imbalance. Use `fake_balanced_gate: true` for benchmarking, real load-balance loss for training. |
| Hang at start (no OOM) | NCCL timeout. Raise `dist_env.timeout_minutes`. |
| Only on multi-node | Rendezvous / firewall. Check `MASTER_ADDR`, `MASTER_PORT`, and that the first node is reachable from all others. |

## When NOT to scale up parallelism

Increasing `tp_size` past intra-node (e.g. tp=8 on a 4-GPU node) requires NVLink between *all* TP ranks — it will still run with PCIe/IB but throughput collapses. Keep TP within a node. Same for EP on small MoEs — if you only have 8 experts and 32 ranks, ep=8 leaves 24 ranks doing pure DP which is fine.

## The one-sentence version

Scale memory first with FSDP2 + activation checkpointing; scale model width with TP + SP within a node; scale model depth with PP across nodes; scale MoE experts with EP; scale sequence length with CP.
