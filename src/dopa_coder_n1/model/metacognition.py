from __future__ import annotations

import torch
from torch import nn


class FailurePredictionGate(nn.Module):
    """Tiny Hot-Core failure predictor used by the metacognition dual gate."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, hot_hidden: torch.Tensor) -> torch.Tensor:
        if hot_hidden.ndim == 3:
            pooled = hot_hidden.mean(dim=1)
        elif hot_hidden.ndim == 2:
            pooled = hot_hidden
        else:
            raise ValueError("hot_hidden must have shape [batch, tokens, d_model] or [batch, d_model]")
        return torch.sigmoid(self.proj(self.norm(pooled))).squeeze(-1)


def should_trigger_posthoc_learning(
    *,
    predicted_failure_probability: torch.Tensor,
    execution_failed: bool,
    overconfidence_threshold: float = 0.20,
) -> bool:
    """Return true when a failed execution contradicts a confident success prediction."""

    if not execution_failed:
        return False
    probability = predicted_failure_probability.detach().float().mean().item()
    return probability < float(overconfidence_threshold)
