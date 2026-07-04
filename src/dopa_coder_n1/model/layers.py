from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from dopa_coder_n1.model.attention_backend import AttentionBackend, build_attention_backend
from dopa_coder_n1.model.kv_cache import LayerKVCache, PackedLayerKV


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if positions is None:
            cos = self.cos[: x.size(-2)]
            sin = self.sin[: x.size(-2)]
        else:
            cos = self.cos.index_select(0, positions.reshape(-1)).view(*positions.shape, -1)
            sin = self.sin.index_select(0, positions.reshape(-1)).view(*positions.shape, -1)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None,
        max_seq_len: int,
        rope_theta: float,
        dropout: float,
        attention_backend: AttentionBackend | None = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        if n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)
        self.dropout = dropout
        self.attention_backend = attention_backend or build_attention_backend("torch")

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | LayerKVCache | PackedLayerKV | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q, positions=positions)
        k = self.rope(k, positions=positions)
        y, new_cache = self.attention_backend.attention(
            q,
            k,
            v,
            kv_cache=kv_cache,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            dropout_p=self.dropout,
            is_training=self.training,
            attn_mask=attn_mask,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq, self.d_model)
        return self.out_proj(y), new_cache


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None,
        max_seq_len: int,
        ffn_multiplier: float,
        rope_theta: float,
        dropout: float,
        attention_backend: AttentionBackend | None = None,
    ):
        super().__init__()
        hidden_dim = int(math.ceil((ffn_multiplier * d_model) / 256) * 256)
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            dropout=dropout,
            attention_backend=attention_backend,
        )
        self.ffn = SwiGLU(d_model, hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | LayerKVCache | PackedLayerKV | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_cache = self.attn(self.attn_norm(x), attn_mask, positions, kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache
