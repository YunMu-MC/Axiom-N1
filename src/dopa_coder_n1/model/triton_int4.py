from __future__ import annotations

from math import prod

import torch

from dopa_coder_n1.model.kv_cache import (
    LayerKVCache,
    PackedKV,
    PackedLayerKV,
    dequantize_int4_tensor,
    unpack_layer_kv,
)

try:  # pragma: no cover - exercised only on CUDA/Triton machines.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU and Windows fallback path.
    triton = None
    tl = None


if triton is not None:  # pragma: no cover - requires Triton runtime.

    @triton.jit
    def _dequant_int4_kernel(
        packed_ptr,
        scale_ptr,
        out_ptr,
        n_elements,
        BLOCK: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        byte_offsets = offsets // 2
        packed = tl.load(packed_ptr + byte_offsets, mask=mask, other=0)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        use_high = (offsets & 1) == 1
        q_u = tl.where(use_high, high, low)
        q = q_u.to(tl.float32) - 8.0
        scale = tl.load(scale_ptr).to(tl.float32)
        tl.store(out_ptr + offsets, q * scale, mask=mask)

else:
    _dequant_int4_kernel = None


def triton_int4_status(device: torch.device | str | None = None) -> dict[str, bool]:
    torch_device = torch.device(device) if device is not None else None
    cuda_device = torch_device is None or torch_device.type == "cuda"
    return {
        "triton_installed": triton is not None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_requested": cuda_device,
        "usable": triton is not None and torch.cuda.is_available() and cuda_device,
    }


def is_triton_int4_available(device: torch.device | str | None = None) -> bool:
    return triton_int4_status(device)["usable"]


def dequantize_packed_kv_triton(
    packed: PackedKV,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    allow_fallback: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize packed int4 K/V with a Triton CUDA kernel when available.

    CPU-only machines and environments without Triton use the portable PyTorch
    reference path. This keeps configs identical across local tests and CUDA
    runs while still giving the final backend a real kernel dispatch point.
    """
    torch_device = torch.device(device)
    if not is_triton_int4_available(torch_device):
        if not allow_fallback:
            raise RuntimeError("Triton int4 kernel is not usable in this environment")
        return _reference_dequantize(packed, device=torch_device, dtype=dtype)
    try:
        k = _dequantize_tensor_triton(
            packed.k,
            packed.k_scale,
            packed.shape,
            device=torch_device,
            dtype=dtype,
        )
        v = _dequantize_tensor_triton(
            packed.v,
            packed.v_scale,
            packed.shape,
            device=torch_device,
            dtype=dtype,
        )
    except Exception:
        if not allow_fallback:
            raise
        return _reference_dequantize(packed, device=torch_device, dtype=dtype)
    return k, v


def unpack_layer_kv_triton(
    packed: PackedLayerKV,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    allow_fallback: bool = True,
) -> LayerKVCache:
    torch_device = torch.device(device)
    if not is_triton_int4_available(torch_device):
        if not allow_fallback:
            raise RuntimeError("Triton int4 kernel is not usable in this environment")
        return unpack_layer_kv(packed, device=torch_device, dtype=dtype)
    try:
        k, v = dequantize_packed_kv_triton(
            packed.window,
            device=torch_device,
            dtype=dtype,
            allow_fallback=allow_fallback,
        )
    except Exception:
        if not allow_fallback:
            raise
        return unpack_layer_kv(packed, device=torch_device, dtype=dtype)
    k_summary = packed.k_summary
    v_summary = packed.v_summary
    if k_summary is not None:
        k_summary = k_summary.to(device=torch_device, dtype=dtype)
    if v_summary is not None:
        v_summary = v_summary.to(device=torch_device, dtype=dtype)
    return LayerKVCache(
        k=k,
        v=v,
        k_summary=k_summary,
        v_summary=v_summary,
        has_summary=packed.has_summary,
    )


def _dequantize_tensor_triton(
    packed: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if _dequant_int4_kernel is None:
        raise RuntimeError("Triton int4 kernel is not available")
    n_elements = prod(shape)
    if n_elements == 0:
        return torch.empty(shape, device=device, dtype=dtype)
    packed_cuda = packed.contiguous().to(device=device, dtype=torch.uint8, non_blocking=True)
    scale_cuda = scale.reshape(()).to(device=device, dtype=torch.float32, non_blocking=True)
    out = torch.empty(n_elements, device=device, dtype=torch.float32)
    block = 256
    grid = (triton.cdiv(n_elements, block),)
    _dequant_int4_kernel[grid](packed_cuda, scale_cuda, out, n_elements, BLOCK=block)
    return out.view(shape).to(dtype=dtype)


def _reference_dequantize(
    packed: PackedKV,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = dequantize_int4_tensor(packed.k, packed.k_scale, packed.shape).to(device=device, dtype=dtype)
    v = dequantize_int4_tensor(packed.v, packed.v_scale, packed.shape).to(device=device, dtype=dtype)
    return k, v
