from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SelfDrivenInductiveLoss(nn.Module):
    """Self-driven inductive training loss from the learning supplement.

    The loss keeps two structural variants tied to the same rule family:
    Variant A should preserve the original skeleton, Variant B should transfer
    the rule under a stronger surface change, and reverse prediction asks the
    model output to recover the minimal rule description.
    """

    def __init__(
        self,
        consistency_weight: float = 0.10,
        transfer_weight: float = 0.10,
        reverse_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.consistency_weight = float(consistency_weight)
        self.transfer_weight = float(transfer_weight)
        self.reverse_weight = float(reverse_weight)

    def forward(
        self,
        original_embedding: torch.Tensor,
        variant_a_embedding: torch.Tensor,
        variant_b_embedding: torch.Tensor,
        *,
        reverse_logits: torch.Tensor | None = None,
        rule_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        consistency = self._one_minus_cosine(original_embedding, variant_a_embedding)
        transfer = self._one_minus_cosine(original_embedding, variant_b_embedding)
        reverse = original_embedding.new_zeros(())
        if reverse_logits is not None and rule_labels is not None:
            reverse = self._reverse_prediction_loss(reverse_logits, rule_labels)

        total = (
            self.consistency_weight * consistency
            + self.transfer_weight * transfer
            + self.reverse_weight * reverse
        )
        return total, {
            "sdit_consistency": consistency,
            "sdit_transfer": transfer,
            "sdit_reverse": reverse,
        }

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.mean(dim=1)
        return x

    @classmethod
    def _one_minus_cosine(cls, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = cls._pool(left).float()
        right = cls._pool(right).float()
        return (1.0 - F.cosine_similarity(left, right, dim=-1, eps=1e-8)).mean()

    @staticmethod
    def _reverse_prediction_loss(reverse_logits: torch.Tensor, rule_labels: torch.Tensor) -> torch.Tensor:
        if reverse_logits.ndim != 3:
            raise ValueError("reverse_logits must have shape [batch, tokens, vocab]")
        if rule_labels.ndim != 2:
            raise ValueError("rule_labels must have shape [batch, tokens]")
        usable = min(reverse_logits.size(1), rule_labels.size(1))
        if usable <= 0:
            return reverse_logits.new_zeros(())
        logits = reverse_logits[:, :usable].contiguous()
        labels = rule_labels[:, :usable].to(logits.device).contiguous()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
