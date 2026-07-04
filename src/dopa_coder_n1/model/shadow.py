from __future__ import annotations

import torch
from torch import nn

from dopa_coder_n1.model.kv_cache import dequantize_int4_tensor, quantize_int4_tensor


class ShadowLinear(nn.Module):
    """Frozen base linear layer plus sparse fake-INT8 trainable shadow delta.

    When `base_quantization="int4"`, the frozen base weight is kept as packed
    signed INT4 buffers instead of an FP parameter. This is the Hot Core path
    closest to the DoAP V2 paper: 4-bit resident base weights and sparse INT8
    shadow deltas carrying gradients.
    """

    def __init__(
        self,
        base: nn.Linear,
        density: float = 0.02,
        mask_strategy: str = "fisher_proxy",
        fake_int8: bool = True,
        base_quantization: str = "fp",
    ):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = base.bias
        self.density = density
        self.mask_strategy = mask_strategy
        self.fake_int8 = fake_int8
        self.base_quantization = base_quantization
        base_weight = base.weight.detach().clone()
        if base_quantization == "int4":
            packed, scale = quantize_int4_tensor(base_weight)
            self.register_buffer("packed_weight", packed)
            self.register_buffer("weight_scale", scale.to(torch.float32))
        else:
            self.weight = nn.Parameter(base_weight, requires_grad=False)
        self.register_buffer("mask", self._new_mask(base_weight))
        self.delta = nn.Parameter(torch.zeros_like(base_weight))
        self.register_buffer("shadow_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("shadow_int8", torch.zeros_like(base_weight, dtype=torch.int8))

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.out_features, self.in_features)

    def base_weight(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        if self.base_quantization == "int4":
            weight = dequantize_int4_tensor(self.packed_weight, self.weight_scale, self.weight_shape)
        else:
            weight = self.weight
        if device is not None or dtype is not None:
            weight = weight.to(device=device, dtype=dtype)
        return weight

    @torch.no_grad()
    def set_base_weight(self, weight: torch.Tensor) -> None:
        if self.base_quantization == "int4":
            packed, scale = quantize_int4_tensor(weight)
            self.packed_weight = packed.to(self.packed_weight.device)
            self.weight_scale = scale.to(self.weight_scale.device, dtype=torch.float32)
        else:
            self.weight.copy_(weight.to(self.weight.device, self.weight.dtype))

    def _new_mask(self, base_weight: torch.Tensor | None = None, avoid: torch.Tensor | None = None) -> torch.Tensor:
        if self.density <= 0:
            return torch.zeros(self.weight_shape, device=self.delta.device if hasattr(self, "delta") else None, dtype=torch.float32)
        if base_weight is None:
            base_weight = self.base_weight(device=self.mask.device if hasattr(self, "mask") else None)
        total = base_weight.numel()
        count = max(1, int(round(total * self.density)))
        if self.mask_strategy == "fisher_proxy" and avoid is None:
            scores = base_weight.detach().abs().flatten()
            idx = torch.topk(scores, k=min(count, total)).indices
            mask = torch.zeros(total, device=base_weight.device, dtype=torch.float32)
            mask[idx] = 1.0
            return mask.view_as(base_weight)
        mask = (torch.rand_like(base_weight) < self.density).to(torch.float32)
        if avoid is not None and torch.equal(mask.to(avoid.device), avoid):
            mask = (torch.rand_like(base_weight) < self.density).to(torch.float32)
        return mask

    def _fake_int8_delta(self) -> torch.Tensor:
        masked = self.delta * self.mask.to(self.delta.device, self.delta.dtype)
        if not self.fake_int8:
            return masked
        max_abs = masked.detach().abs().max().clamp_min(1e-8)
        scale = max_abs / 127.0
        q = torch.clamp(torch.round(masked.detach() / scale), -127, 127).to(torch.int8)
        with torch.no_grad():
            self.shadow_scale.copy_(scale.detach().to(self.shadow_scale.device, dtype=torch.float32))
            self.shadow_int8.copy_(q.to(self.shadow_int8.device))
        dequant = q.to(masked.dtype) * scale.to(masked.dtype)
        return masked + (dequant - masked).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.base_weight(device=x.device, dtype=x.dtype) + self._fake_int8_delta().to(x.device, x.dtype)
        bias = self.bias.to(x.device, x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, weight, bias)

    @torch.no_grad()
    def rotate_mask(self, preserve_delta: bool = True) -> None:
        if preserve_delta:
            new_base = self.base_weight(device=self.delta.device, dtype=self.delta.dtype) + self._fake_int8_delta()
            self.set_base_weight(new_base)
        self.delta.zero_()
        self.mask.copy_(self._new_mask(avoid=self.mask).to(self.mask.device))
        self.shadow_int8.zero_()
        self.shadow_scale.fill_(1.0)


def inject_shadow_linears(
    module: nn.Module,
    density: float,
    name_filter: tuple[str, ...] = ("w", "proj"),
    mask_strategy: str = "fisher_proxy",
    fake_int8: bool = True,
    base_quantization: str = "fp",
) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(token in name for token in name_filter):
            setattr(
                module,
                name,
                ShadowLinear(
                    child,
                    density=density,
                    mask_strategy=mask_strategy,
                    fake_int8=fake_int8,
                    base_quantization=base_quantization,
                ),
            )
            replaced += 1
        else:
            replaced += inject_shadow_linears(
                child,
                density=density,
                name_filter=name_filter,
                mask_strategy=mask_strategy,
                fake_int8=fake_int8,
                base_quantization=base_quantization,
            )
    return replaced


def inject_dopa_shadow_linears(
    model: nn.Module,
    hot_density: float,
    cold_density: float,
    mask_strategy: str = "fisher_proxy",
    fake_int8: bool = True,
    hot_base_quantization: str = "int4",
    cold_base_quantization: str = "fp",
) -> int:
    replaced = 0
    hot_layers = getattr(model, "hot_layers", None)
    if hot_layers is not None and hot_density > 0:
        replaced += inject_shadow_linears(
            hot_layers,
            density=hot_density,
            mask_strategy=mask_strategy,
            fake_int8=fake_int8,
            base_quantization=hot_base_quantization,
        )
    cold_manager = getattr(model, "cold_manager", None)
    if cold_manager is not None and cold_density > 0:
        replaced += inject_shadow_linears(
            cold_manager,
            density=cold_density,
            mask_strategy=mask_strategy,
            fake_int8=fake_int8,
            base_quantization=cold_base_quantization,
        )
    fine_cold_shell = getattr(model, "fine_cold_shell", None)
    if fine_cold_shell is not None and cold_density > 0:
        store = getattr(fine_cold_shell, "store", None)
        if store is not None:
            store.shadow_density = cold_density
        replaced += inject_shadow_linears(
            fine_cold_shell,
            density=cold_density,
            mask_strategy=mask_strategy,
            fake_int8=fake_int8,
            base_quantization=cold_base_quantization,
        )
    return replaced


def rotate_shadow_masks(module: nn.Module, preserve_delta: bool = True) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, ShadowLinear):
            child.rotate_mask(preserve_delta=preserve_delta)
            count += 1
    return count
