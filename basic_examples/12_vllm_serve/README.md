# Module 12 — vLLM serving

Automodel's consolidated checkpoints are HF-compatible, so vLLM can serve them directly. This module stands up an OpenAI-compatible HTTP server on a compute node, pointed at either a Hub model or a checkpoint you trained in Modules 02–04, and hits it with a client.

## Why not `model.generate`?

`transformers.model.generate()` (Module 05) is fine for single-prompt demos and batch offline inference. vLLM adds:

- **Continuous batching** — requests are packed across the token dimension, not the batch dimension. 5–20× higher throughput at typical chat workloads.
- **PagedAttention** — KV-cache paged like OS virtual memory, so many concurrent sessions fit in GPU memory.
- **OpenAI-compatible HTTP API** — any client that speaks `/v1/chat/completions` works unchanged.
- **Streaming token delivery** — SSE out of the box.

Rule of thumb: one interactive user → `model.generate` is enough. Multiple users / evals with many prompts / production serving → vLLM.

## One-time setup

vLLM isn't in the container. Install into a writable overlay:

```bash
bash 12_vllm_serve/install_overlay.sh         # writes 12_vllm_serve/overlay.img (~8 GB)
```

The installer pulls vLLM 0.11.x (`--no-deps` to avoid downgrading torch) plus the runtime extras it expects (FastAPI, ray, xformers, openai SDK, etc.). Verifies with `python -c "import vllm; print(vllm.__version__)"`.

## Run

```bash
# Smoke: serve Qwen3-0.6B + fire a prompt, then shut down.
sbatch 12_vllm_serve/serve.slrm

# Serve your LoRA-merged checkpoint from Module 04:
MODEL=$CKPT_ROOT/trackB_qwen3_0p6b_lora_squad/epoch_0_step_299/model/consolidated \
    sbatch 12_vllm_serve/serve.slrm

# Custom model from Module 11 (needs the RoPE-GPT config class registered — see gotchas):
MODEL=$CKPT_ROOT/custom_rope_gpt/epoch_0_step_499/model/consolidated \
    sbatch 12_vllm_serve/serve.slrm
```

The sbatch starts `vllm serve`, waits up to 120 s for `/v1/models` to 200-OK, fires a single client query via `client_example.py`, then shuts down.

## Long-running server + external clients

The `serve.sh` wrapper binds `0.0.0.0:8000`, so a client on another node can hit `http://<srun-node>:8000/v1`. Inside a Slurm job:

```bash
# Find the node's hostname:
echo "http://$(hostname):8000/v1"

# Keep the server up until the Slurm alloc ends:
OVERLAY=12_vllm_serve/overlay.img 12_vllm_serve/serve.sh Qwen/Qwen3-0.6B 8000 --max-model-len 4096
```

In another terminal (on any node that can reach the Slurm net):

```bash
shared/launch.sh python 12_vllm_serve/client_example.py \
    --endpoint http://<node>:8000/v1 \
    --model Qwen/Qwen3-0.6B \
    --prompt "List three sustainable energy sources."
```

Streaming:
```bash
shared/launch.sh python 12_vllm_serve/client_example.py \
    --endpoint http://<node>:8000/v1 \
    --model Qwen/Qwen3-0.6B --prompt "Write a haiku." --stream
```

## curl-level smoke

If you want to rule out the SDK:

```bash
curl http://<node>:8000/v1/models

curl http://<node>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50
  }'
```

## Gotchas

- **Cold start is slow.** vLLM's first request on a freshly-loaded model can take 60–120 s (graph compile + CUDA graph capture). The sbatch's readiness loop allows 120 s. Raise it if you're serving bigger models.
- **`--max-model-len`** must be ≤ the model's trained context. Qwen3-0.6B's limit is 32768; setting `--max-model-len 2048` keeps KV memory small on 1×H100.
- **Custom-model registration is not automatic in vLLM.** Our Module 11 `rope_gpt` type needs to be recognized by vLLM's model loader. For a stock HF `PreTrainedModel` subclass, the simplest path is `--trust-remote-code` + make sure `custom_model.py` is on `PYTHONPATH` before `vllm serve` is invoked. More rigorously, vLLM has a `MODEL_REGISTRY` and you'd need a [custom model registration](https://docs.vllm.ai/en/latest/contributing/model/basic.html). Out of scope here; use Module 05's `generate.py` for custom models until you commit the architecture to vLLM's registry.
- **LoRA adapters:** vLLM supports them natively (`--enable-lora`, `--lora-modules name=/path`). For this workshop we merge-consolidate in Module 04 via `is_peft: true` + `save_consolidated: true`, so serving the merged checkpoint is a normal `vllm serve <dir>` call with no LoRA flags.
- **Memory accounting:** `--gpu-memory-utilization 0.9` (default) leaves 10 % headroom. If you land on a shared node, drop to 0.6 so other jobs don't OOM you (or request exclusive GPUs).
- **torch version drift:** vLLM releases pin torch tightly. The overlay installer uses `--no-deps` to avoid downgrading the container's torch; if import fails at runtime, the fallback is to rerun `install_overlay.sh` without `--no-deps`, which will install vLLM's pinned torch *on top of* the container's torch (last-wins). That occasionally breaks other Automodel-side imports — use a new overlay if this matters.

## Perf sanity check

```bash
# Throughput benchmark baked into vLLM:
OVERLAY=12_vllm_serve/overlay.img shared/launch.sh \
    vllm bench throughput --model Qwen/Qwen3-0.6B \
        --num-prompts 200 --input-len 256 --output-len 256
```

Expected ballpark on 1× H100 for Qwen3-0.6B:
- First-token latency: ~30–50 ms
- Steady-state throughput: >3k tokens/s aggregate (batched) vs ~100–200 tokens/s for `model.generate` in a loop.

## Related

- Module 04 — produces the consolidated HF checkpoints this module serves.
- Module 05 — classic `model.generate` for single-prompt and offline batch.
- Module 07 — `lm-eval-harness` can be pointed at a vLLM endpoint via `--model vllm --model_args base_url=http://...`, but we use the HF path there for simplicity.
