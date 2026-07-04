from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from dopa_coder_n1.model.quantization import dequantize_int4_state_dict, dequantize_int8_state_dict


class ColdBlock(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x, **kwargs)
        return x


class ColdBlockManager(nn.Module):
    """Cold shell manager with CPU/GPU residency hooks.

    Blocks can live on CPU and be moved to the active device on demand. For real 64B
    training, pass checkpoint_dir and store each block as block_{i}.pt to avoid keeping
    all cold weights in GPU memory.
    """

    def __init__(
        self,
        blocks: list[ColdBlock | None],
        block_factory: Callable[[int], ColdBlock] | None = None,
        num_blocks: int | None = None,
        cold_device: str = "cpu",
        active_device: str = "cuda",
        max_active: int = 2,
        checkpoint_dir: str | None = None,
        lazy: bool = False,
        storage_dtype: str = "fp16",
    ):
        super().__init__()
        self.lazy = lazy
        self.block_factory = block_factory
        self.num_blocks = num_blocks if num_blocks is not None else len(blocks)
        self.blocks = nn.ModuleList([b for b in blocks if b is not None])
        self._lazy_blocks: dict[int, ColdBlock] = {}
        self.cold_device = torch.device(cold_device)
        self.active_device = torch.device(active_device if torch.cuda.is_available() else "cpu")
        self.max_active = max_active
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.storage_dtype = storage_dtype
        self._lru: OrderedDict[int, None] = OrderedDict()
        for block in self.blocks:
            block.to(self.cold_device)

    def offload_all(self) -> None:
        self._lru.clear()
        for block in self.blocks:
            block.to(self.cold_device)
        for block in self._lazy_blocks.values():
            block.to(self.cold_device)

    def materialize(self, index: int, active_device: torch.device | str | None = None) -> ColdBlock:
        target_device = torch.device(active_device) if active_device is not None else self.active_device
        block = self._get_block(index)
        self.load_block_weights(index, block)
        if target_device.type == "cuda" and self.storage_dtype in {"fp16", "float16"}:
            block.to(dtype=torch.float16)
        elif target_device.type == "cuda" and self.storage_dtype in {"bf16", "bfloat16"}:
            block.to(dtype=torch.bfloat16)
        else:
            block.to(dtype=torch.float32)
        block.to(target_device)
        self._lru[index] = None
        self._lru.move_to_end(index)
        while len(self._lru) > self.max_active:
            old, _ = self._lru.popitem(last=False)
            self._get_block(old).to(self.cold_device)
        return block

    def load_block_weights(self, index: int, block: ColdBlock | None = None) -> ColdBlock:
        block = self._get_block(index) if block is None else block
        if self.checkpoint_dir is not None:
            ckpt = self.checkpoint_dir / f"block_{index}.pt"
            q4ckpt = self.checkpoint_dir / f"block_{index}.int4.pt"
            qckpt = self.checkpoint_dir / f"block_{index}.int8.pt"
            if ckpt.exists():
                block.load_state_dict(torch.load(ckpt, map_location=self.cold_device))
            elif q4ckpt.exists():
                packed = torch.load(q4ckpt, map_location="cpu")
                block.load_state_dict(dequantize_int4_state_dict(packed))
            elif qckpt.exists():
                packed = torch.load(qckpt, map_location="cpu")
                block.load_state_dict(dequantize_int8_state_dict(packed))
        return block

    def _get_block(self, index: int) -> ColdBlock:
        if self.lazy:
            if index not in self._lazy_blocks:
                if self.block_factory is None:
                    raise RuntimeError("lazy cold block requested without block_factory")
                block = self.block_factory(index)
                block.to(self.cold_device)
                self._lazy_blocks[index] = block
                self.add_module(f"lazy_block_{index}", block)
            return self._lazy_blocks[index]
        return self.blocks[index]

    def forward_selected(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.num_blocks == 0:
            return torch.zeros_like(x)
        out = torch.zeros_like(x)
        reduce_dims = tuple(range(weights.ndim - 1))
        selected = torch.nonzero(weights.detach().sum(dim=reduce_dims) > 0, as_tuple=False).flatten().tolist()
        active_device = x.device
        for idx in selected:
            block = self.materialize(int(idx), active_device=active_device)
            block_dtype = next(block.parameters()).dtype
            block_kwargs = _move_tensor_kwargs(kwargs, device=active_device)
            block_out = block(x.to(device=active_device, dtype=block_dtype), **block_kwargs)
            if weights.ndim == 3:
                w = weights[..., idx].to(block_out.device, block_out.dtype).unsqueeze(-1)
            else:
                w = weights[:, idx].to(block_out.device, block_out.dtype).view(-1, 1, 1)
            out = out.to(block_out.device) + w * block_out
        return out


def _move_tensor_kwargs(kwargs: dict, *, device: torch.device) -> dict:
    moved = {}
    for key, value in kwargs.items():
        moved[key] = value.to(device=device) if isinstance(value, torch.Tensor) else value
    return moved
