from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ExternalKnowledgeSupervision:
    labels: torch.Tensor
    mask: torch.Tensor
    hot_token_losses: torch.Tensor | None = None
    teacher_token_losses: torch.Tensor | None = None


def derive_external_knowledge_labels(
    hot_token_losses: torch.Tensor,
    teacher_token_losses: torch.Tensor,
    *,
    hot_threshold: float,
    teacher_easy_threshold: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label tokens where the hot core struggles but the teacher predicts easily."""
    hot = hot_token_losses.float()
    teacher = teacher_token_losses.to(hot.device, hot.dtype)
    valid = torch.isfinite(hot) & torch.isfinite(teacher)
    need_external = (
        valid
        & (hot >= hot_threshold)
        & (teacher <= teacher_easy_threshold)
        & ((hot - teacher) >= margin)
    )
    return need_external.to(hot.dtype), valid


def shifted_token_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-position next-token CE aligned to the hidden state that predicts it."""
    target = labels[:, 1:].to(logits.device)
    pred = logits[:, :-1]
    losses = F.cross_entropy(
        pred.reshape(-1, pred.size(-1)),
        target.reshape(-1),
        reduction="none",
    ).view(target.shape)
    return losses


def align_token_values(
    values: torch.Tensor,
    shape: torch.Size | tuple[int, int],
    *,
    fill_value: float,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    rows, cols = int(shape[0]), int(shape[1])
    out_dtype = dtype or values.dtype
    out_device = device or values.device
    values = values.to(out_device, out_dtype)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.size(0) == 1 and rows > 1:
        values = values.expand(rows, -1)
    if values.size(0) != rows:
        raise ValueError(f"token supervision batch mismatch: expected {rows}, got {values.size(0)}")
    if values.size(1) == cols:
        return values
    if values.size(1) > cols:
        return values[:, :cols]
    pad = torch.full((rows, cols - values.size(1)), fill_value, device=out_device, dtype=out_dtype)
    return torch.cat([values, pad], dim=1)


def resolve_external_knowledge_supervision(batch: dict, aux: dict, cfg) -> ExternalKnowledgeSupervision | None:
    difficulty = aux.get("difficulty")
    if difficulty is None:
        return None
    shape = difficulty.shape
    device = difficulty.device
    dtype = difficulty.dtype
    if "external_knowledge_labels" in batch:
        raw = align_token_values(
            batch["external_knowledge_labels"],
            shape,
            fill_value=-1.0,
            dtype=dtype,
            device=device,
        )
        mask = torch.isfinite(raw) & (raw >= 0)
        return ExternalKnowledgeSupervision(labels=raw.clamp(0.0, 1.0), mask=mask)
    teacher_raw = batch.get("teacher_token_losses")
    if teacher_raw is None:
        return None
    teacher = align_token_values(teacher_raw, shape, fill_value=float("nan"), dtype=dtype, device=device)
    hot_raw = batch.get("hot_token_losses")
    if hot_raw is not None:
        hot = align_token_values(hot_raw, shape, fill_value=float("nan"), dtype=dtype, device=device)
    else:
        hot_logits = aux.get("hot_logits")
        labels = batch.get("labels")
        if hot_logits is None or labels is None:
            return None
        hot = align_token_values(
            shifted_token_losses(hot_logits, labels.to(hot_logits.device)),
            shape,
            fill_value=float("nan"),
            dtype=dtype,
            device=device,
        )
    labels, mask = derive_external_knowledge_labels(
        hot,
        teacher,
        hot_threshold=cfg.dopa.external_knowledge_hot_loss_threshold,
        teacher_easy_threshold=cfg.dopa.external_knowledge_teacher_easy_threshold,
        margin=cfg.dopa.external_knowledge_margin,
    )
    return ExternalKnowledgeSupervision(labels=labels, mask=mask, hot_token_losses=hot, teacher_token_losses=teacher)


def external_knowledge_orchestration_loss(batch: dict, aux: dict, cfg) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    supervision = resolve_external_knowledge_supervision(batch, aux, cfg)
    if supervision is None or not bool(supervision.mask.any()):
        return None, {}
    difficulty = aux["difficulty"].to(supervision.labels.device, supervision.labels.dtype)
    target = supervision.labels[supervision.mask]
    gate_pred = difficulty[supervision.mask].clamp(1e-6, 1.0 - 1e-6)
    with torch.autocast(device_type=gate_pred.device.type, enabled=False):
        gate_bce = F.binary_cross_entropy(gate_pred.float(), target.float())
    loss = cfg.dopa.external_knowledge_gate_loss_weight * gate_bce
    metrics = {
        "external_knowledge_bce": gate_bce.detach(),
        "external_knowledge_positive_rate": target.mean().detach(),
    }
    cold_logits = aux.get("cold_logits")
    if cold_logits is not None and cfg.dopa.external_knowledge_ldp_loss_weight > 0:
        need_logits = cold_logits.max(dim=-1).values
        need_logits = align_token_values(
            need_logits,
            supervision.labels.shape,
            fill_value=float("nan"),
            dtype=supervision.labels.dtype,
            device=supervision.labels.device,
        )
        with torch.autocast(device_type=need_logits.device.type, enabled=False):
            ldp_bce = F.binary_cross_entropy_with_logits(need_logits[supervision.mask].float(), target.float())
        loss = loss + cfg.dopa.external_knowledge_ldp_loss_weight * ldp_bce
        metrics["ldp_need_bce"] = ldp_bce.detach()
    if supervision.hot_token_losses is not None and supervision.teacher_token_losses is not None:
        margin = (supervision.hot_token_losses - supervision.teacher_token_losses)[supervision.mask]
        if margin.numel() > 0:
            metrics["hot_teacher_loss_margin"] = margin.mean().detach()
    return loss, metrics
