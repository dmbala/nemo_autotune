# Module 01 — Data processing

Three sub-tasks, each producing an artifact a later module consumes:

| Script | Output | Used by |
|---|---|---|
| `tiny_shakespeare.py` | `data/shakespeare/{train,val}.bin` + `.bos.idx` | Module 02 tiny-scale pretrain |
| `fineweb_10bt.sh` | `data/fineweb_500M/*.bin` + `.bos.idx` | Module 02 124M pretrain, Module 03 distributed |
| `jsonl_to_chat.py` | `data/sft/*.jsonl` | Module 04 SFT |

## Tiny path (laptop-scale, ~30 s)

```bash
shared/launch.sh python 01_data/tiny_shakespeare.py --out data/shakespeare
```

Downloads Tiny-Shakespeare (~1 MB), BPE-tokenizes with GPT-2, writes a `uint16` binary shard + a `.bos.idx` sidecar. Magic/version/layout match `/opt/Automodel/tools/nanogpt_data_processor.py` exactly, so `NanogptDataset` reads it without special-casing.

## FineWeb path (cluster-scale, ~20–60 min on 32 CPUs)

```bash
sbatch 01_data/run_fineweb.slrm
# or directly: 01_data/fineweb_10bt.sh $DATA_ROOT/fineweb_500M 500M
```

Wraps the upstream `tools/nanogpt_data_processor.py`. Downloads the HuggingFaceFW/fineweb `sample-10BT` split and tokenizes up to `max_tokens` (default 500M) into GPT-2 `uint16` shards.

Scale knob: pass a different `max_tokens` (`100M`, `1B`, `2B`). Each shard caps at ~4 GB; the script rotates files automatically.

## Chat JSONL (for SFT)

```bash
shared/launch.sh python 01_data/jsonl_to_chat.py \
    --dataset rajpurkar/squad --split train \
    --out data/sft/squad_train.jsonl --max-samples 5000

shared/launch.sh python 01_data/jsonl_to_chat.py \
    --dataset rajpurkar/squad --split validation \
    --out data/sft/squad_val.jsonl --max-samples 500
```

The JSONL schema matches `nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset`: one `{"messages": [...]}` object per line with `role` ∈ `{user, assistant}`.

If you want to finetune on SQuAD *without* chat-style formatting, Module 04 Track A uses the repo's native `make_squad_dataset` loader directly — no JSONL needed.

## Schema note — NanogptDataset file format

From `/opt/Automodel/nemo_automodel/components/datasets/llm/nanogpt_dataset.py`:

```
int32[256] header:
    [0] magic = 2788_95051   (new) or 20240520 (legacy fineweb.py)
    [1] version = 1
    [2] num_tokens
    [3] itemsize (2 for uint16, 4 for uint32)
uint16/uint32[num_tokens] tokens
<name>.bos.idx (sidecar): int32[] byte positions of BOS tokens
```

`tiny_shakespeare.py` replicates this inline (~30 lines) so the format is visible in the workshop source. The larger `fineweb_10bt.sh` just calls the upstream tool.

## Verification

```bash
shared/launch.sh python -c "
from nemo_automodel.components.datasets.llm.nanogpt_dataset import NanogptDataset
ds = NanogptDataset(file_pattern='data/shakespeare/train.bin', seq_len=256)
ex = next(iter(ds))
print('keys:', list(ex.keys()))
print('input_ids[:8]:', list(ex['input_ids'][:8]))
"
```
Expect `keys: ['input_ids', 'labels']` and a reasonable token sequence. The dataset yields lists that the default collator turns into padded batch tensors.
