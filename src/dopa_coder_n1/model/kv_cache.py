from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch


def quantize_int4_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack signed int4 values into uint8 plus one global scale.

    This is a portable reference implementation. Production kernels should use
    per-head scales and fused dequantization.
    """
    max_abs = x.detach().abs().max().clamp_min(1e-8)
    scale = max_abs / 7.0
    q = torch.clamp(torch.round(x.detach().cpu() / scale.cpu()), -8, 7).to(torch.int8)
    q_u = (q + 8).to(torch.uint8).flatten()
    if q_u.numel() % 2:
        q_u = torch.cat([q_u, torch.zeros(1, dtype=torch.uint8)])
    packed = (q_u[0::2] & 0x0F) | ((q_u[1::2] & 0x0F) << 4)
    return packed, scale.cpu()


def dequantize_int4_tensor(packed: torch.Tensor, scale: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.empty(packed.numel() * 2, dtype=torch.int8, device=packed.device)
    q[0::2] = low.to(torch.int8) - 8
    q[1::2] = high.to(torch.int8) - 8
    q = q[: prod(shape)].view(shape)
    return q.to(torch.float32) * scale.to(device=packed.device, dtype=torch.float32)


@dataclass
class PackedKV:
    k: torch.Tensor
    v: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    shape: tuple[int, ...]


@dataclass
class LayerKVCache:
    k: torch.Tensor
    v: torch.Tensor
    k_summary: torch.Tensor | None = None
    v_summary: torch.Tensor | None = None
    has_summary: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.k.shape)

    def attention_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.has_summary and self.k_summary is not None and self.v_summary is not None:
            return torch.cat([self.k_summary, self.k], dim=2), torch.cat([self.v_summary, self.v], dim=2)
        return self.k, self.v


@dataclass
class PackedLayerKV:
    window: PackedKV
    k_summary: torch.Tensor | None = None
    v_summary: torch.Tensor | None = None
    has_summary: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return self.window.shape


def _as_layer_kv(cache: tuple[torch.Tensor, torch.Tensor] | LayerKVCache) -> LayerKVCache:
    if isinstance(cache, LayerKVCache):
        return cache
    k, v = cache
    return LayerKVCache(k=k, v=v)


def pack_layer_kv(cache: tuple[torch.Tensor, torch.Tensor] | LayerKVCache) -> PackedLayerKV:
    cache = _as_layer_kv(cache)
    k, v = cache.k, cache.v
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError("KV tensors must have the same shape before int4 packing")
    k_pack, k_scale = quantize_int4_tensor(k)
    v_pack, v_scale = quantize_int4_tensor(v)
    return PackedLayerKV(
        window=PackedKV(k=k_pack, v=v_pack, k_scale=k_scale, v_scale=v_scale, shape=tuple(k.shape)),
        k_summary=cache.k_summary.detach().cpu() if cache.k_summary is not None else None,
        v_summary=cache.v_summary.detach().cpu() if cache.v_summary is not None else None,
        has_summary=cache.has_summary,
    )


def unpack_layer_kv(
    packed: PackedLayerKV,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> LayerKVCache:
    k = dequantize_int4_tensor(packed.window.k, packed.window.k_scale, packed.window.shape)
    v = dequantize_int4_tensor(packed.window.v, packed.window.v_scale, packed.window.shape)
    k_summary = packed.k_summary
    v_summary = packed.v_summary
    if device is not None or dtype is not None:
        k = k.to(device=device, dtype=dtype)
        v = v.to(device=device, dtype=dtype)
        if k_summary is not None:
            k_summary = k_summary.to(device=device, dtype=dtype)
        if v_summary is not None:
            v_summary = v_summary.to(device=device, dtype=dtype)
    return LayerKVCache(
        k=k,
        v=v,
        k_summary=k_summary,
        v_summary=v_summary,
        has_summary=packed.has_summary,
    )


class HotKVCache:
    """Legacy flat-vector reference cache.

    Runtime incremental decoding uses LayerKVCache/PackedLayerKV below because
    attention needs head-shaped tensors.
    """

    def __init__(self, layers: int, window: int, d_model: int, decay: float = 0.99):
        self.layers = layers
        self.window = window
        self.d_model = d_model
        self.decay = decay
        self.items: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in range(layers)]
        self.k_summary = [torch.zeros(d_model) for _ in range(layers)]
        self.v_summary = [torch.zeros(d_model) for _ in range(layers)]

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        k = k.detach().cpu().reshape(-1)
        v = v.detach().cpu().reshape(-1)
        bucket = self.items[layer]
        bucket.append((k, v))
        if len(bucket) > self.window:
            old_k, old_v = bucket.pop(0)
            self.k_summary[layer] = self.decay * self.k_summary[layer] + (1.0 - self.decay) * old_k
            self.v_summary[layer] = self.decay * self.v_summary[layer] + (1.0 - self.decay) * old_v

    def get_dense(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bucket = self.items[layer]
        if bucket:
            k = torch.stack([x[0] for x in bucket], dim=0)
            v = torch.stack([x[1] for x in bucket], dim=0)
        else:
            k = torch.empty(0, self.d_model)
            v = torch.empty(0, self.d_model)
        return self.k_summary[layer], self.v_summary[layer], k, v

    def packed_window(self, layer: int) -> PackedKV:
        _, _, k, v = self.get_dense(layer)
        k_pack, k_scale = quantize_int4_tensor(k)
        v_pack, v_scale = quantize_int4_tensor(v)
        return PackedKV(k_pack, v_pack, k_scale, v_scale, tuple(k.shape))


class ColdSelectiveKVCache:
    """Small 4-bit cache for frequently activated cold attention head units."""

    def __init__(self, max_units: int, window: int, head_dim: int):
        self.max_units = max_units
        self.window = window
        self.head_dim = head_dim
        self.cache: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.priority: list[str] = []

    def mark_hot(self, unit_key: str) -> None:
        if unit_key not in self.cache:
            if len(self.cache) >= self.max_units:
                evict = self.priority.pop(0)
                self.cache.pop(evict, None)
            self.cache[unit_key] = []
            self.priority.append(unit_key)

    def append(self, unit_key: str, k: torch.Tensor, v: torch.Tensor) -> None:
        if unit_key not in self.cache:
            return
        bucket = self.cache[unit_key]
        bucket.append((k.detach().cpu().reshape(-1), v.detach().cpu().reshape(-1)))
        if len(bucket) > self.window:
            bucket.pop(0)

    def packed(self, unit_key: str) -> PackedKV | None:
        bucket = self.cache.get(unit_key)
        if not bucket:
            return None
        k = torch.stack([x[0] for x in bucket], dim=0)
        v = torch.stack([x[1] for x in bucket], dim=0)
        k_pack, k_scale = quantize_int4_tensor(k)
        v_pack, v_scale = quantize_int4_tensor(v)
        return PackedKV(k_pack, v_pack, k_scale, v_scale, tuple(k.shape))


class ColdSelectiveKVState:
    """Runtime packed cache for selected cold attention head units."""

    def __init__(self, max_units: int, window: int):
        self.max_units = max_units
        self.window = window
        self.cache: dict[str, PackedLayerKV] = {}
        self.priority: list[str] = []

    def get(
        self,
        unit_key: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> LayerKVCache | None:
        packed = self.cache.get(unit_key)
        if packed is None:
            return None
        self._touch(unit_key)
        return unpack_layer_kv(packed, device=device, dtype=dtype)

    def get_packed(self, unit_key: str) -> PackedLayerKV | None:
        packed = self.cache.get(unit_key)
        if packed is not None:
            self._touch(unit_key)
        return packed

    def put(self, unit_key: str, cache: tuple[torch.Tensor, torch.Tensor] | LayerKVCache) -> None:
        if self.max_units <= 0 or self.window <= 0:
            return
        layer_cache = _as_layer_kv(cache)
        if layer_cache.k.size(2) > self.window:
            layer_cache = LayerKVCache(
                k=layer_cache.k[:, :, -self.window :].contiguous(),
                v=layer_cache.v[:, :, -self.window :].contiguous(),
                k_summary=layer_cache.k_summary,
                v_summary=layer_cache.v_summary,
                has_summary=layer_cache.has_summary,
            )
        self._touch(unit_key)
        while len(self.priority) > self.max_units:
            evict = self.priority.pop(0)
            self.cache.pop(evict, None)
        self.cache[unit_key] = pack_layer_kv(layer_cache)

    def _touch(self, unit_key: str) -> None:
        if unit_key in self.priority:
            self.priority.remove(unit_key)
        self.priority.append(unit_key)
