from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class AlignmentGateResult:
    score: torch.Tensor
    triggered: torch.Tensor
    intent_vector: torch.Tensor
    implementation_vector: torch.Tensor


class IntentImplementationAlignmentGate(nn.Module):
    """Intent-Implementation Alignment Gate.

    The gate compares a pooled user-intent vector against a pooled
    implementation vector. When the score drops below the threshold, callers
    can softly suppress tokens marked by `token_drift_bias`.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        *,
        threshold: float = 0.30,
        soft_strength: float = 1.0,
        window_tokens: int = 512,
        task_type_count: int = 2,
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.soft_strength = float(soft_strength)
        self.window_tokens = int(window_tokens)
        self.scorer = nn.Linear(2 * d_model, 1)
        self.task_type_embedding = nn.Embedding(max(1, task_type_count), d_model)
        self.register_buffer("token_drift_bias", torch.zeros(vocab_size, dtype=torch.float32))

    def forward(
        self,
        intent_hidden: torch.Tensor,
        implementation_hidden: torch.Tensor,
        *,
        task_type_ids: torch.Tensor | None = None,
    ) -> AlignmentGateResult:
        intent = self._pool(intent_hidden)
        implementation = self._pool(implementation_hidden, tail_window=self.window_tokens)
        if task_type_ids is not None:
            task_type_ids = task_type_ids.to(intent.device).clamp_min(0)
            task_type_ids = task_type_ids % self.task_type_embedding.num_embeddings
            intent = intent + self.task_type_embedding(task_type_ids)
        if intent.size(0) == 1 and implementation.size(0) > 1:
            intent = intent.expand(implementation.size(0), -1)
        elif implementation.size(0) == 1 and intent.size(0) > 1:
            implementation = implementation.expand(intent.size(0), -1)
        score = torch.sigmoid(self.scorer(torch.cat([intent, implementation], dim=-1))).squeeze(-1)
        return AlignmentGateResult(
            score=score,
            triggered=score < self.threshold,
            intent_vector=intent,
            implementation_vector=implementation,
        )

    def apply_logit_bias(self, logits: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        if self.token_drift_bias.abs().sum() == 0:
            return logits
        original_shape = logits.shape
        if logits.ndim == 3:
            batch = logits.size(0)
            score = score.view(batch, 1, 1)
            bias = self.token_drift_bias.to(logits.device, logits.dtype).view(1, 1, -1)
        elif logits.ndim == 2:
            batch = logits.size(0)
            score = score.view(batch, 1)
            bias = self.token_drift_bias.to(logits.device, logits.dtype).view(1, -1)
        else:
            raise ValueError(f"logits must be [B,V] or [B,T,V], got {original_shape}")
        penalty = (self.threshold - score.to(logits.dtype)).clamp_min(0.0) * self.soft_strength
        return logits - penalty * bias

    def set_token_drift_bias(self, values: dict[int, float]) -> None:
        self.token_drift_bias.zero_()
        for token_id, value in values.items():
            if 0 <= int(token_id) < self.token_drift_bias.numel():
                self.token_drift_bias[int(token_id)] = float(value)

    @staticmethod
    def _pool(hidden: torch.Tensor, tail_window: int | None = None) -> torch.Tensor:
        if hidden.ndim == 2:
            return hidden
        if hidden.ndim != 3:
            raise ValueError("hidden must be [B,T,D] or [B,D]")
        if tail_window is not None and tail_window > 0:
            hidden = hidden[:, -tail_window:, :]
        return hidden.mean(dim=1)
