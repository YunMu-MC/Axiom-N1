from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DSparkDraftOutput:
    draft_logits: torch.Tensor
    markov_logits: torch.Tensor
    corrected_logits: torch.Tensor
    confidence: torch.Tensor
    draft_tokens: torch.Tensor


@dataclass
class VerificationSchedule:
    verify_lengths: torch.Tensor
    mask: torch.Tensor
    prefix_survival: torch.Tensor


@dataclass
class AcceptanceResult:
    accepted_lengths: torch.Tensor
    accepted_tokens: list[torch.Tensor]


class DSparkHeads(nn.Module):
    """Semi-autoregressive DSpark heads mounted on the Hot Core."""

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        gamma: int = 7,
        markov_rank: int = 8,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if gamma < 2:
            raise ValueError("DSpark gamma must be at least 2")
        self.gamma = int(gamma)
        self.draft_positions = self.gamma - 1
        self.vocab_size = int(vocab_size)
        hidden = hidden_dim or d_model
        self.parallel_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.draft_positions * vocab_size),
        )
        self.markov_left = nn.Embedding(vocab_size, markov_rank)
        self.markov_right = nn.Linear(markov_rank, vocab_size, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.draft_positions, vocab_size))
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.draft_positions),
        )

    def forward(self, anchor_hidden: torch.Tensor, *, previous_tokens: torch.Tensor) -> DSparkDraftOutput:
        if anchor_hidden.ndim != 2:
            raise ValueError("anchor_hidden must have shape [batch, d_model]")
        batch = anchor_hidden.size(0)
        draft_logits = self.parallel_head(anchor_hidden).view(batch, self.draft_positions, self.vocab_size)
        markov_seed = self.markov_right(self.markov_left(previous_tokens.view(-1)))
        markov_logits = markov_seed.unsqueeze(1).expand(batch, self.draft_positions, self.vocab_size)
        corrected_logits = draft_logits + markov_logits + self.position_bias.unsqueeze(0)
        confidence = torch.sigmoid(self.confidence_head(anchor_hidden))
        draft_tokens = corrected_logits.argmax(dim=-1)
        return DSparkDraftOutput(
            draft_logits=draft_logits,
            markov_logits=markov_logits,
            corrected_logits=corrected_logits,
            confidence=confidence,
            draft_tokens=draft_tokens,
        )


class VerificationScheduler:
    """Confidence-scheduled verification length selector."""

    def __init__(
        self,
        *,
        gamma: int = 7,
        min_verify_tokens: int = 1,
        low_load_threshold: float = 0.20,
        high_load_threshold: float = 0.70,
    ) -> None:
        self.gamma = int(gamma)
        self.max_positions = self.gamma - 1
        self.min_verify_tokens = int(min_verify_tokens)
        self.low_load_threshold = float(low_load_threshold)
        self.high_load_threshold = float(high_load_threshold)

    def __call__(self, confidence: torch.Tensor, *, engine_load: float | torch.Tensor = 0.0) -> VerificationSchedule:
        confidence = confidence.clamp(0.0, 1.0)
        prefix_survival = confidence.cumprod(dim=-1)
        if isinstance(engine_load, torch.Tensor):
            load = float(engine_load.detach().float().mean().item())
        else:
            load = float(engine_load)
        load = max(0.0, min(1.0, load))
        threshold = self.low_load_threshold + (self.high_load_threshold - self.low_load_threshold) * load
        keep = prefix_survival >= threshold
        lengths = keep.long().sum(dim=-1).clamp(min=self.min_verify_tokens, max=self.max_positions)
        positions = torch.arange(self.max_positions, device=confidence.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(-1)
        return VerificationSchedule(verify_lengths=lengths, mask=mask, prefix_survival=prefix_survival)


def accept_prefix_from_distributions(
    draft_tokens: torch.Tensor,
    draft_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
) -> AcceptanceResult:
    """Deterministic prefix acceptance used by the local DSpark path.

    Full stochastic rejection sampling is reserved for production sampling. For
    tests and greedy inference, accepting while the target argmax agrees gives a
    stable no-quality-loss verification boundary.
    """

    target_tokens = target_log_probs.argmax(dim=-1)
    matches = target_tokens.eq(draft_tokens)
    lengths = []
    accepted = []
    for row, row_matches in zip(draft_tokens, matches):
        length = 0
        for ok in row_matches.tolist():
            if not ok:
                break
            length += 1
        lengths.append(length)
        accepted.append(row[:length].detach().clone())
    return AcceptanceResult(
        accepted_lengths=torch.tensor(lengths, device=draft_tokens.device, dtype=torch.long),
        accepted_tokens=accepted,
    )
