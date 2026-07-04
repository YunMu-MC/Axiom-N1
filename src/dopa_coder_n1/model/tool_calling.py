from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class ToolCallingOutput:
    need_logits: torch.Tensor
    argument_validity: torch.Tensor
    query_embedding: torch.Tensor


class ToolCallingHeads(nn.Module):
    """Small heads that train the model to decide, retrieve, and validate tool calls."""

    def __init__(self, d_model: int, action_count: int, query_dim: int):
        super().__init__()
        self.need_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, action_count))
        self.argument_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.query_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, query_dim))

    def forward(self, hidden: torch.Tensor) -> ToolCallingOutput:
        pooled = hidden[:, -1]
        return ToolCallingOutput(
            need_logits=self.need_head(pooled),
            argument_validity=torch.sigmoid(self.argument_head(pooled)).squeeze(-1),
            query_embedding=F.normalize(self.query_head(pooled), dim=-1),
        )
