from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F

from dopa_coder_n1.model.kv_cache import LayerKVCache, PackedLayerKV, unpack_layer_kv

AttentionBackendName = Literal["torch", "int4_reference", "triton_int4"]
VALID_ATTENTION_BACKENDS = {"torch", "int4_reference", "triton_int4"}


@dataclass(frozen=True)
class AttentionBackendConfig:
    name: AttentionBackendName = "torch"


class AttentionBackend:
    """Backend boundary for dense and packed-KV attention.

    `int4_reference` accepts PackedLayerKV and dequantizes with portable PyTorch.
    `triton_int4` uses a Triton CUDA kernel for packed int4 KV dequantization
    when available, then calls PyTorch SDPA. CPU-only environments fall back to
    the same reference path so model code and configs stay portable.
    """

    def __init__(self, config: AttentionBackendConfig | None = None):
        self.config = config or AttentionBackendConfig()

    def decode_cache(
        self,
        cache: tuple[torch.Tensor, torch.Tensor] | LayerKVCache | PackedLayerKV | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | LayerKVCache | None:
        if isinstance(cache, PackedLayerKV):
            if self.config.name == "triton_int4":
                from dopa_coder_n1.model.triton_int4 import unpack_layer_kv_triton

                return unpack_layer_kv_triton(cache, device=device, dtype=dtype)
            return unpack_layer_kv(cache, device=device, dtype=dtype)
        return cache

    def attention(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        *,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | LayerKVCache | PackedLayerKV | None,
        n_heads: int,
        n_kv_heads: int,
        dropout_p: float,
        is_training: bool,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        cache = self.decode_cache(kv_cache, device=q.device, dtype=q.dtype)
        has_cache = cache is not None
        if cache is not None:
            if isinstance(cache, LayerKVCache):
                window_k, window_v = cache.k, cache.v
                attn_k, attn_v = cache.attention_kv()
            else:
                window_k, window_v = cache
                attn_k, attn_v = window_k, window_v
            new_cache = (torch.cat([window_k, k_new], dim=2), torch.cat([window_v, v_new], dim=2))
            k = torch.cat([attn_k, k_new], dim=2)
            v = torch.cat([attn_v, v_new], dim=2)
        else:
            new_cache = (k_new, v_new)
            k = k_new
            v = v_new
        if n_kv_heads != n_heads:
            repeat = n_heads // n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p if is_training else 0.0,
            is_causal=attn_mask is None and not has_cache,
        )
        return y, new_cache


def build_attention_backend(name: str) -> AttentionBackend:
    if name not in VALID_ATTENTION_BACKENDS:
        raise ValueError(f"unknown attention backend: {name}")
    return AttentionBackend(AttentionBackendConfig(name=name))  # type: ignore[arg-type]
