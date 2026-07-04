from __future__ import annotations

import torch

from dopa_coder_n1.model.kv_cache import dequantize_int4_tensor, quantize_int4_tensor


def quantize_int8_state_dict(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
    packed: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {}
    for name, tensor in state.items():
        if not torch.is_floating_point(tensor):
            packed[name] = tensor
            continue
        max_abs = tensor.detach().abs().max().clamp_min(1e-8)
        scale = max_abs / 127.0
        q = torch.clamp(torch.round(tensor.detach().cpu() / scale.cpu()), -127, 127).to(torch.int8)
        packed[name] = {"q": q, "scale": scale.cpu(), "shape": torch.tensor(tensor.shape)}
    return packed


def dequantize_int8_state_dict(packed: dict) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in packed.items():
        if isinstance(value, dict) and "q" in value:
            state[name] = value["q"].to(torch.float32) * value["scale"].to(torch.float32)
        else:
            state[name] = value
    return state


def quantize_int4_state_dict(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
    packed: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {}
    for name, tensor in state.items():
        if not torch.is_floating_point(tensor):
            packed[name] = tensor
            continue
        q, scale = quantize_int4_tensor(tensor)
        packed[name] = {"q4": q, "scale": scale, "shape": tuple(tensor.shape)}
    return packed


def dequantize_int4_state_dict(packed: dict) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in packed.items():
        if isinstance(value, dict) and "q4" in value:
            shape = value["shape"]
            if isinstance(shape, torch.Tensor):
                shape = tuple(int(x) for x in shape.tolist())
            state[name] = dequantize_int4_tensor(value["q4"], value["scale"], tuple(shape))
        else:
            state[name] = value
    return state
