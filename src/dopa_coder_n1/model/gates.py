from __future__ import annotations

import torch
from torch import nn

from dopa_coder_n1.model.lora_bank import sparse_topk_softmax


def _active_mean(hidden: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
    if hidden.ndim == 2:
        return hidden
    if active is None:
        return hidden.mean(dim=1)
    weights = active.to(hidden.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


class DifficultyGate(nn.Module):
    """Token-level difficulty gate.

    DoAP V2 routes hard tokens through the Cold Shell. The previous engineering
    version pooled the last token; this version returns one score per token.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(hidden)).squeeze(-1)


class LayerDemandPredictor(nn.Module):
    def __init__(self, d_model: int, blocks: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, blocks),
        )

    def forward(self, hidden: torch.Tensor, active: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(hidden)
        weights = sparse_topk_softmax(logits, self.top_k)
        if active is not None:
            weights = weights * active.to(weights.dtype).unsqueeze(-1)
        return weights, logits


class HotColdFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model))

    def forward(self, hot: torch.Tensor, cold: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([hot, cold], dim=-1)))
        return g * cold + (1.0 - g) * hot


class CuriosityGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        pooled = _active_mean(hidden)
        return torch.sigmoid(self.net(pooled)).squeeze(-1)
