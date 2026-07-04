from __future__ import annotations

import torch
from torch import nn


def sparse_topk_softmax(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.size(-1))
    values, indices = torch.topk(scores, k=k, dim=-1)
    weights = torch.softmax(values, dim=-1).to(dtype=scores.dtype)
    out = scores.new_zeros(scores.shape)
    return out.scatter(-1, indices, weights)


class LoRABank(nn.Module):
    """Variable-rank LoRA bank with multiple target sites.

    The paper describes one LoRA module as covering several Hot Core projection
    sites instead of a single hidden-space adapter. For this compact runtime we
    keep one hidden injection point, but the bank owns per-site A/B factors and
    averages their deltas. Rank gates make each module variable-rank under a
    differentiable relaxation.
    """

    def __init__(
        self,
        modules: int,
        d_model: int,
        rank: int,
        alpha: float,
        target_sites: int = 1,
    ):
        super().__init__()
        self.modules = modules
        self.rank = rank
        self.target_sites = max(1, target_sites)
        self.scaling = alpha / rank
        self.a = nn.Parameter(torch.empty(modules, self.target_sites, d_model, rank))
        self.b = nn.Parameter(torch.empty(modules, self.target_sites, rank, d_model))
        self.rank_logits = nn.Parameter(torch.zeros(modules, rank))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        nn.init.zeros_(self.b)

    def rank_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.rank_logits)

    def active_ranks(self, threshold: float = 0.5) -> torch.Tensor:
        return (self.rank_gates() >= threshold).sum(dim=-1).clamp_min(1)

    def forward(self, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], coeffs: [B, M]
        xa = torch.einsum("btd,msdr->btmsr", x, self.a)
        xa = xa * self.rank_gates().to(dtype=xa.dtype).view(1, 1, self.modules, 1, self.rank)
        xab = torch.einsum("btmsr,msrd->btmsd", xa, self.b)
        site_mixed = xab.mean(dim=3)
        mixed = torch.einsum("bm,btmd->btd", coeffs.to(site_mixed.dtype), site_mixed)
        return mixed * self.scaling


class HyperNetwork(nn.Module):
    def __init__(self, skeleton_dim: int, hidden_dim: int, lora_modules: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.net = nn.Sequential(
            nn.Linear(skeleton_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, lora_modules),
        )

    def forward(self, skeleton_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.net(skeleton_embedding)
        coeffs = sparse_topk_softmax(scores, self.top_k)
        return coeffs, scores
