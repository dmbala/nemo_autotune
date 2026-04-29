"""A minimal custom GPT variant with RoPE positional embeddings.

Demonstrates how to plug a user-defined model into Automodel's recipe via the
YAML ``model._target_`` hook.

Automodel's checkpointing path (``base_recipe.py:353``) unconditionally calls
``model.save_pretrained(...)``, so any custom model must be a
``transformers.PreTrainedModel``. That also gives you
``AutoModelForCausalLM.from_pretrained(<consolidated_dir>)`` for free at
inference time (Module 05). The pattern:

  1. Subclass ``PretrainedConfig`` for the hyperparams.
  2. Subclass ``PreTrainedModel`` for the module.
  3. Reference them from YAML via the standard
     ``NeMoAutoModelForCausalLM.from_config`` builder with
     ``_target_: <your.module.YourConfig>``.

Usage (from a YAML):

    model:
      _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_config
      config:
        _target_: custom_model.RoPEGPTConfig
        vocab_size: 50258
        n_positions: 512
        n_embd: 192
        n_layer: 6
        n_head: 4

    # Make this file importable:   PYTHONPATH=<dir containing this file>.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


# ---------- RoPE ----------
def _rope_freqs(seq_len: int, head_dim: int, device, base: float = 10_000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)


# ---------- Layers ----------
class RoPEAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_rope(q, cos[:T], sin[:T])
        k = _apply_rope(k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class RoPEBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = RoPEAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


# ---------- HF-registered config & model ----------
class RoPEGPTConfig(PretrainedConfig):
    """HuggingFace config for the custom RoPE-GPT."""

    model_type = "rope_gpt"

    def __init__(
        self,
        vocab_size: int = 50258,
        n_positions: int = 512,
        n_embd: int = 192,
        n_layer: int = 6,
        n_head: int = 4,
        dropout: float = 0.0,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class RoPEGPTForCausalLM(PreTrainedModel):
    """Custom RoPE-GPT wired into the HF ecosystem.

    Inherits ``save_pretrained`` / ``from_pretrained`` from ``PreTrainedModel``
    so Automodel's consolidated-checkpoint path works and Module 05 inference
    can load us via ``AutoModelForCausalLM.from_pretrained(...)``.
    """

    config_class = RoPEGPTConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False

    def __init__(self, config: RoPEGPTConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [RoPEBlock(config.n_embd, config.n_head, config.dropout) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tie
        head_dim = config.n_embd // config.n_head
        cos, sin = _rope_freqs(config.n_positions, head_dim, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def get_input_embeddings(self) -> nn.Module:
        return self.wte

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.wte = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        **kwargs,  # absorb attention_mask, position_ids, etc.
    ) -> CausalLMOutput:
        B, T = input_ids.shape
        if T > self.config.n_positions:
            raise ValueError(f"seq_len {T} exceeds n_positions {self.config.n_positions}")

        x = self.wte(input_ids)
        cos = self.rope_cos.to(x.device, dtype=x.dtype)
        sin = self.rope_sin.to(x.device, dtype=x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits)


# Optional: make transformers' AutoConfig/AutoModel aware of the new type so
# `AutoModelForCausalLM.from_pretrained(<ckpt>)` works without extra args in
# downstream inference code.
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

AutoConfig.register("rope_gpt", RoPEGPTConfig)
AutoModelForCausalLM.register(RoPEGPTConfig, RoPEGPTForCausalLM)
