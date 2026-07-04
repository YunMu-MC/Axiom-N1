from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from dopa_coder_n1.model.attention_backend import AttentionBackend, build_attention_backend
from dopa_coder_n1.model.layers import RMSNorm, RotaryEmbedding
from dopa_coder_n1.model.kv_cache import ColdSelectiveKVState, LayerKVCache, PackedLayerKV
from dopa_coder_n1.model.quantization import dequantize_int4_state_dict, dequantize_int8_state_dict


@dataclass(frozen=True)
class ColdUnitId:
    layer: int
    kind: str
    index: int

    @property
    def key(self) -> str:
        return f"L{self.layer:03d}_{self.kind}_{self.index:03d}"


@dataclass
class ColdUnitSelection:
    units: list[ColdUnitId]
    weights: torch.Tensor
    logits: torch.Tensor


class AttentionHeadUnit(nn.Module):
    """One cold-shell attention head unit.

    It owns q/k/v projections for one head plus that head's output projection slice.
    This matches the DoAP V2 supplement's SSD load granularity.
    """

    def __init__(
        self,
        d_model: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        dropout: float,
        attention_backend: AttentionBackend | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.out_proj = nn.Linear(head_dim, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, theta=rope_theta)
        self.dropout = dropout
        self.attention_backend = attention_backend or build_attention_backend("torch")

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        kv_cache: LayerKVCache | PackedLayerKV | None = None,
        return_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h = self.norm(x)
        q = self.q_proj(h).unsqueeze(1)
        k = self.k_proj(h).unsqueeze(1)
        v = self.v_proj(h).unsqueeze(1)
        q = self.rope(q, positions=positions)
        k = self.rope(k, positions=positions)
        y, new_cache = self.attention_backend.attention(
            q,
            k,
            v,
            kv_cache=kv_cache,
            n_heads=1,
            n_kv_heads=1,
            dropout_p=self.dropout,
            is_training=self.training,
        )
        y = y.squeeze(1)
        out = self.out_proj(y)
        if return_kv_cache:
            return out, new_cache
        return out


class FFNSubBlockUnit(nn.Module):
    """One FFN segment W_in[j] -> activation -> W_out[j]."""

    def __init__(self, d_model: int, segment_dim: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.w_in = nn.Linear(d_model, segment_dim, bias=False)
        self.w_out = nn.Linear(segment_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        del positions
        h = self.norm(x)
        return self.w_out(self.dropout(F.gelu(self.w_in(h))))


class ColdUnitStore(nn.Module):
    """Lazy SSD-backed store for fine-grained cold units."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        ffn_multiplier: float,
        ffn_subblocks: int,
        rope_theta: float,
        dropout: float,
        checkpoint_dir: str | None,
        cold_device: str,
        active_device: str,
        storage_dtype: str = "fp16",
        shadow_density: float = 0.0,
        attention_backend: AttentionBackend | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.ffn_hidden = int(math.ceil((ffn_multiplier * d_model) / ffn_subblocks) * ffn_subblocks)
        self.ffn_subblocks = ffn_subblocks
        self.segment_dim = self.ffn_hidden // ffn_subblocks
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.cold_device = torch.device(cold_device)
        self.active_device = torch.device(active_device if torch.cuda.is_available() else "cpu")
        self.storage_dtype = storage_dtype
        self.shadow_density = shadow_density
        self.attention_backend = attention_backend or build_attention_backend("torch")
        self._cache: dict[str, nn.Module] = {}

    def create(self, unit_id: ColdUnitId) -> nn.Module:
        if unit_id.kind == "head":
            unit = AttentionHeadUnit(
                self.d_model,
                self.head_dim,
                max_seq_len=self.max_seq_len,
                rope_theta=self.rope_theta,
                dropout=self.dropout,
                attention_backend=self.attention_backend,
            )
        elif unit_id.kind == "ffn":
            unit = FFNSubBlockUnit(self.d_model, self.segment_dim, dropout=self.dropout)
        else:
            raise ValueError(f"unknown cold unit kind: {unit_id.kind}")
        if self.shadow_density > 0:
            from dopa_coder_n1.model.shadow import inject_shadow_linears

            inject_shadow_linears(unit, density=self.shadow_density)
        return unit

    def load(self, unit_id: ColdUnitId) -> nn.Module:
        key = unit_id.key
        if key not in self._cache:
            unit = self.create(unit_id)
            self._load_weights(unit_id, unit)
            unit.to(self.cold_device)
            self._cache[key] = unit
            self.add_module(f"unit_{key}", unit)
        unit = self._cache[key]
        if self.active_device.type == "cuda" and self.storage_dtype in {"fp16", "float16"}:
            unit.to(device=self.active_device, dtype=torch.float16)
        elif self.active_device.type == "cuda" and self.storage_dtype in {"bf16", "bfloat16"}:
            unit.to(device=self.active_device, dtype=torch.bfloat16)
        else:
            unit.to(device=self.active_device, dtype=torch.float32)
        return unit

    def _load_weights(self, unit_id: ColdUnitId, unit: nn.Module) -> None:
        if self.checkpoint_dir is None:
            return
        path = self.checkpoint_dir / f"{unit_id.key}.pt"
        int4_path = self.checkpoint_dir / f"{unit_id.key}.int4.pt"
        int8_path = self.checkpoint_dir / f"{unit_id.key}.int8.pt"
        if path.exists():
            unit.load_state_dict(_fill_missing_shadow_state(unit, torch.load(path, map_location=self.cold_device)))
        elif int4_path.exists():
            state = dequantize_int4_state_dict(torch.load(int4_path, map_location="cpu"))
            unit.load_state_dict(_fill_missing_shadow_state(unit, state))
        elif int8_path.exists():
            state = dequantize_int8_state_dict(torch.load(int8_path, map_location="cpu"))
            unit.load_state_dict(_fill_missing_shadow_state(unit, state))

    def offload(self, unit_id: ColdUnitId) -> None:
        unit = self._cache.get(unit_id.key)
        if unit is not None:
            unit.to(self.cold_device)


def _fill_missing_shadow_state(module: nn.Module, state: dict) -> dict:
    current = module.state_dict()
    if all(key in state for key in current):
        return state
    patched = dict(state)
    for key, value in current.items():
        if key in patched:
            continue
        if (
            key.endswith(".delta")
            or key.endswith(".mask")
            or key.endswith(".shadow_scale")
            or key.endswith(".shadow_int8")
            or key.endswith(".packed_weight")
            or key.endswith(".weight_scale")
        ):
            patched[key] = value
    return patched


class FineGrainedLayerDemandPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        cold_layers: int,
        n_heads: int,
        ffn_subblocks: int,
        layer_budget: int,
        head_budget: int,
        ffn_budget: int,
        coverage_penalty: float = 0.0,
    ):
        super().__init__()
        self.cold_layers = cold_layers
        self.n_heads = n_heads
        self.ffn_subblocks = ffn_subblocks
        self.layer_budget = layer_budget
        self.head_budget = head_budget
        self.ffn_budget = ffn_budget
        self.coverage_penalty = coverage_penalty
        self.total_units = cold_layers * (n_heads + ffn_subblocks)
        self.register_buffer("visit_counts", torch.zeros(self.total_units), persistent=False)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.total_units),
        )

    def decode_index(self, flat: int) -> ColdUnitId:
        per_layer = self.n_heads + self.ffn_subblocks
        layer = flat // per_layer
        offset = flat % per_layer
        if offset < self.n_heads:
            return ColdUnitId(layer=layer, kind="head", index=offset)
        return ColdUnitId(layer=layer, kind="ffn", index=offset - self.n_heads)

    def forward(self, hidden: torch.Tensor, active: torch.Tensor | None = None) -> ColdUnitSelection:
        logits = self.net(hidden)
        route_logits = logits
        if active is not None:
            mask_value = torch.finfo(route_logits.dtype).min
            route_logits = torch.where(active.unsqueeze(-1), route_logits, torch.full_like(route_logits, mask_value))
        if self.coverage_penalty > 0:
            coverage = self.visit_counts.to(logits.device, logits.dtype)
            coverage = coverage / coverage.max().clamp_min(1.0)
            route_logits = route_logits - self.coverage_penalty * coverage.view(*([1] * (route_logits.ndim - 1)), -1)
        per_layer = self.n_heads + self.ffn_subblocks
        if route_logits.ndim == 3:
            if active is not None:
                active_f = active.to(route_logits.dtype).unsqueeze(-1)
                mean_logits = (route_logits.masked_fill(~active.unsqueeze(-1), 0.0) * active_f).sum(dim=(0, 1))
                mean_logits = mean_logits / active_f.sum().clamp_min(1.0)
            else:
                mean_logits = route_logits.mean(dim=(0, 1))
        else:
            mean_logits = route_logits.mean(dim=0)
        layer_scores = mean_logits.view(self.cold_layers, per_layer).max(dim=1).values
        layer_k = min(max(1, self.layer_budget), self.cold_layers)
        _, layer_pos = torch.topk(layer_scores, k=layer_k)
        selected_layers = sorted(int(i) for i in layer_pos)
        head_indices = [
            layer * per_layer + head
            for layer in selected_layers
            for head in range(self.n_heads)
        ]
        ffn_indices = [
            layer * per_layer + self.n_heads + block
            for layer in selected_layers
            for block in range(self.ffn_subblocks)
        ]
        head_scores = mean_logits[head_indices]
        ffn_scores = mean_logits[ffn_indices]
        head_k = min(self.head_budget, len(head_indices))
        ffn_k = min(self.ffn_budget, len(ffn_indices))
        h_vals, h_pos = torch.topk(head_scores, k=head_k)
        f_vals, f_pos = torch.topk(ffn_scores, k=ffn_k)
        flat_selected = [head_indices[int(i)] for i in h_pos.reshape(-1)] + [
            ffn_indices[int(i)] for i in f_pos.reshape(-1)
        ]
        score_selected = torch.cat([h_vals.reshape(-1), f_vals.reshape(-1)], dim=0)
        weights = torch.softmax(score_selected, dim=0)
        units = [self.decode_index(i) for i in flat_selected]
        self._record_visits(flat_selected)
        return ColdUnitSelection(units=units, weights=weights, logits=logits)

    @torch.no_grad()
    def _record_visits(self, flat_indices: list[int]) -> None:
        if not flat_indices:
            return
        idx = torch.tensor(flat_indices, device=self.visit_counts.device, dtype=torch.long)
        self.visit_counts.index_add_(0, idx, torch.ones_like(idx, dtype=self.visit_counts.dtype))


class FineGrainedColdShell(nn.Module):
    """Fine-grained head/FFN sub-block cold shell with SSD lazy loading."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cold_layers: int,
        max_seq_len: int,
        ffn_multiplier: float,
        ffn_subblocks: int,
        rope_theta: float,
        dropout: float,
        checkpoint_dir: str | None,
        cold_device: str,
        active_device: str,
        storage_dtype: str,
        shadow_density: float = 0.0,
        attention_backend: AttentionBackend | None = None,
    ):
        super().__init__()
        self.cold_layers = cold_layers
        self.n_heads = n_heads
        self.ffn_subblocks = ffn_subblocks
        self.store = ColdUnitStore(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            ffn_multiplier=ffn_multiplier,
            ffn_subblocks=ffn_subblocks,
            rope_theta=rope_theta,
            dropout=dropout,
            checkpoint_dir=checkpoint_dir,
            cold_device=cold_device,
            active_device=active_device,
            storage_dtype=storage_dtype,
            shadow_density=shadow_density,
            attention_backend=attention_backend,
        )

    @property
    def total_units(self) -> int:
        return self.cold_layers * (self.n_heads + self.ffn_subblocks)

    def forward(
        self,
        hidden: torch.Tensor,
        selection: ColdUnitSelection,
        positions: torch.Tensor | None = None,
        cold_kv_state: ColdSelectiveKVState | None = None,
    ) -> torch.Tensor:
        self.store.active_device = hidden.device
        if not selection.units:
            return torch.zeros_like(hidden)
        residual_root = hidden
        state = hidden
        layer_groups: dict[int, list[tuple[torch.Tensor, ColdUnitId]]] = {}
        for weight, unit_id in zip(selection.weights, selection.units):
            layer_groups.setdefault(unit_id.layer, []).append((weight, unit_id))
        for layer in sorted(layer_groups):
            state = self._run_sparse_layer(state, layer_groups[layer], positions, cold_kv_state)
        return (state.to(residual_root.device) - residual_root).to(residual_root.dtype)

    def _run_sparse_layer(
        self,
        hidden: torch.Tensor,
        weighted_units: list[tuple[torch.Tensor, ColdUnitId]],
        positions: torch.Tensor | None,
        cold_kv_state: ColdSelectiveKVState | None,
    ) -> torch.Tensor:
        state = hidden
        head_units = [(w, u) for w, u in weighted_units if u.kind == "head"]
        ffn_units = [(w, u) for w, u in weighted_units if u.kind == "ffn"]
        if head_units:
            attn_delta = self._run_unit_group(state, head_units, positions, cold_kv_state)
            state = state.to(attn_delta.device) + attn_delta
        if ffn_units:
            ffn_delta = self._run_unit_group(state, ffn_units, positions, cold_kv_state=None)
            state = state.to(ffn_delta.device) + ffn_delta
        return state

    def _run_unit_group(
        self,
        hidden: torch.Tensor,
        weighted_units: list[tuple[torch.Tensor, ColdUnitId]],
        positions: torch.Tensor | None,
        cold_kv_state: ColdSelectiveKVState | None,
    ) -> torch.Tensor:
        out: torch.Tensor | None = None
        weight_total = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        for weight, unit_id in weighted_units:
            unit = self.store.load(unit_id)
            unit_hidden = hidden.to(next(unit.parameters()).device)
            unit_weight = weight.to(unit_hidden.device, unit_hidden.dtype)
            if (
                cold_kv_state is not None
                and unit_id.kind == "head"
                and unit_hidden.size(1) == 1
                and isinstance(unit, AttentionHeadUnit)
            ):
                cached = cold_kv_state.get_packed(unit_id.key)
                unit_out, updated = unit(
                    unit_hidden,
                    positions=positions,
                    kv_cache=cached,
                    return_kv_cache=True,
                )
                cold_kv_state.put(unit_id.key, updated)
            else:
                unit_out = unit(unit_hidden, positions=positions)
            out = unit_weight * unit_out if out is None else out.to(unit_out.device) + unit_weight * unit_out
            weight_total = weight_total.to(unit_weight.device) + unit_weight
            if not torch.is_grad_enabled():
                self.store.offload(unit_id)
        if out is None:
            return torch.zeros_like(hidden)
        return out / weight_total.clamp_min(1e-6)
