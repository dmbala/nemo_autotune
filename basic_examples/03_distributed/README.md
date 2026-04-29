# Module 03 — Distributed training & checkpointing

Explores the parallelism knobs that Automodel wires under the YAML `distributed.*` block, and the checkpoint format (PyTorch Distributed Checkpoint, a.k.a. DCP) with its async and consolidated-HF-export options.

## Strategies

All three strategies are selected by a single YAML field:

```yaml
distributed:
  strategy: fsdp2 | megatron_fsdp | ddp
```

| Strategy | When to use | Key fields |
|---|---|---|
| **`fsdp2`** | Default. DTensor-based, composes with TP/PP/CP/EP. | `dp_size`, `dp_replicate_size`, `tp_size`, `pp_size`, `sequence_parallel`, `activation_checkpointing` |
| **`megatron_fsdp`** | When you need Megatron-LM style ZeRO-3 with custom overlap policies. | `zero_dp_strategy`, `overlap_grad_reduce`, `overlap_param_gather` |
| **`ddp`** | Baseline; no sharding, no TP/PP. | `activation_checkpointing`, `backend` |

Full schema: `/opt/Automodel/nemo_automodel/components/distributed/config.py` (dataclasses `FSDP2Config`, `MegatronFSDPConfig`, `DDPConfig`). Parallelism sizes live at the top level of `distributed`, strategy-specific kwargs live inside the dataclass.

### Dimensions

`dp_size` (sharded), `dp_replicate_size` (replicated), `tp_size`, `pp_size`, `cp_size` (sequence / context), `ep_size` (expert MoE).

`dp_size: none` auto-computes as `world_size / (tp_size*pp_size*cp_size*ep_size*dp_replicate_size)`.

## Configs in this module

| Config | Layout | World size | Use |
|---|---|---|---|
| `fsdp2_dp4.yaml` | `dp_size=4` | 4 (1 node) | Baseline single-node FSDP2 |
| `ckpt_async_consolidated.yaml` | `dp_size=4` + `checkpoint.is_async=true` | 4 (1 node) | Demonstrates non-blocking DCP save |
| `fsdp2_multinode_2x4.yaml` | `dp_replicate_size=2` + `dp_size=4` (HSDP) | 8 (2 nodes × 4) | Cross-node replication + intra-node sharding |

Tensor-parallel + pipeline-parallel configs are not included because custom `GPT2Config` models don't ship with a `tp_plan`. See `/opt/Automodel/examples/benchmark/configs/qwen3_moe_30b_torch.yaml` for a TP/EP-aware configuration on a production-scale model.

## Runs

Requires FineWeb-500M data from Module 01 (`sbatch 01_data/run_fineweb.slrm`).

```bash
# Single node, FSDP2 dp=4 (default config):
sbatch 03_distributed/run_fsdp2_single.slrm

# Same job but exercise async DCP + HF consolidation:
CONFIG=ckpt_async_consolidated.yaml sbatch 03_distributed/run_fsdp2_single.slrm

# 2 nodes, HSDP (2×4):
sbatch 03_distributed/run_hsdp_multinode.slrm
```

## Checkpoint format (DCP)

Automodel uses **`torch.distributed.checkpoint`** with per-rank sharded safetensors. Save path: `nemo_automodel/components/checkpoint/checkpointing.py::dcp.save` (line 667). Load path: same file at line 622.

Directory layout per checkpoint step (`epoch_<N>_step_<S>/`):

```
model/
  shard-0000N-model-00001-of-00001.safetensors  # DCP shards, one per rank
  consolidated/                                  # rank-0 writes this when save_consolidated=true
    config.json
    generation_config.json
    model-00001-of-00001.safetensors
    model.safetensors.index.json
optim/    # sharded optimizer state (DCP)
rng/
dataloader/
step_scheduler.pt
config.yaml
losses.json
```

- **DCP shards** are what you resume from. They preserve the exact parallel layout.
- **`consolidated/`** is a plain `save_pretrained` directory — `AutoModelForCausalLM.from_pretrained(consolidated_dir)` works directly. Module 05 inference and Module 07 eval use this.

### Config knobs (dataclass `CheckpointingConfig`)

```yaml
checkpoint:
  enabled: true
  checkpoint_dir: /abs/path            # root dir; epoch_<N>_step_<S>/ is auto-appended
  model_save_format: safetensors        # safetensors | torch_save
  save_consolidated: true               # also emit HF-loadable model/consolidated/
  is_async: true                        # DCP staging path, non-blocking save (torch ≥ 2.9)
  is_peft: false                        # adapter-only save for LoRA runs (Module 04)
  single_rank_consolidation: false      # rank 0 does all the stitching (for remote FS)
  staging_dir: null                     # optional tmpdir for large consolidations
```

## Exporting to HF (after the fact)

If a run was launched without `save_consolidated: true`, use `export_to_hf.py`:

```bash
shared/launch.sh python 03_distributed/export_to_hf.py \
    --in  $CKPT_ROOT/gpt2_124m/epoch_0_step_100/model \
    --out $CKPT_ROOT/gpt2_124m/epoch_0_step_100/model/consolidated
```

## Verification

```bash
# 1. Multi-GPU loss parity with single-GPU (first 100 steps should track within noise):
grep 'step 99' logs/ws02_pretrain_124m.*.out logs/ws03_fsdp2_dp4.*.out

# 2. Checkpoint is HF-loadable:
shared/launch.sh python -c "
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('$CKPT_ROOT/fsdp2_fsdp2_dp4/epoch_0_step_99/model/consolidated')
print('params:', sum(p.numel() for p in m.parameters())/1e6, 'M')
"

# 3. Multi-node rendezvous worked:
grep 'rank' logs/ws03_hsdp_2x4.*.out | head
```

## Gotchas

- `dp_size: none` means "auto" (a string, not Python None). The recipe resolves it from world_size.
- The multi-node sbatch invokes `torchrun` explicitly instead of the `automodel` CLI because torchrun needs to know `--node-rank` per task — that's `SLURM_NODEID`. The CLI wrapper doesn't pass it through.
- HSDP with mismatched `dp_replicate_size * dp_size ≠ world_size` will crash at mesh setup. Always check `dp_replicate_size × dp_size × tp_size × pp_size × cp_size × ep_size == world_size`.
- Async DCP requires `torch>=2.9`; older torch silently downgrades to sync and logs a warning.
