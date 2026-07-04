from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
from torch import nn
from torch.nn import functional as F


DELIBERATION_LEVELS = ("low", "medium", "high", "ultra", "x_open")


@dataclass
class DeliberationDecision:
    logits: torch.Tensor
    complexity_score: torch.Tensor
    confidence: torch.Tensor
    selected_level: torch.Tensor


def parse_requested_deliberation_level(text: str | None) -> int | None:
    if not text:
        return None
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text.lower())
    if re.search(r"\bx\s*open\b|xopen|开放|無限制|无限制", normalized):
        return 4
    if "ultra" in normalized or "超高" in normalized or "全面调研" in normalized:
        return 3
    if "deep" in normalized or "high" in normalized or "深思" in normalized or "仔细" in normalized:
        return 2
    if "medium" in normalized or "中等" in normalized or "自检" in normalized:
        return 1
    if "low" in normalized or "fast" in normalized or "快速" in normalized or "简短" in normalized:
        return 0
    return None


def should_escalate_level(
    current_level: int,
    *,
    verification_failed: bool = False,
    new_information: bool = False,
    max_level: int = 4,
) -> int:
    if verification_failed or new_information:
        return min(max_level, int(current_level) + 1)
    return int(current_level)


class AdaptiveDeliberationScheduler(nn.Module):
    """Five-level metacognitive scheduler for adaptive deliberation."""

    def __init__(self, d_model: int, hidden_dim: int = 128, level_count: int = 5) -> None:
        super().__init__()
        self.level_count = int(level_count)
        self.complexity = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.policy = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, level_count),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        curiosity_confidence: torch.Tensor,
        failure_probability: torch.Tensor | None = None,
        task_complexity: torch.Tensor | None = None,
        requested_level: str | int | torch.Tensor | None = None,
    ) -> DeliberationDecision:
        pooled = self._pool(hidden)
        learned_complexity = torch.sigmoid(self.complexity(pooled)).squeeze(-1)
        if task_complexity is not None:
            complexity_score = task_complexity.to(pooled.device, pooled.dtype).view(-1).clamp(0.0, 1.0)
        else:
            complexity_score = learned_complexity
        if failure_probability is None:
            failure_probability = torch.zeros_like(complexity_score)
        failure_probability = failure_probability.to(pooled.device, pooled.dtype).view(-1).clamp(0.0, 1.0)
        curiosity_confidence = curiosity_confidence.to(pooled.device, pooled.dtype).view(-1).clamp(0.0, 1.0)
        confidence = 1.0 - ((1.0 - curiosity_confidence) + failure_probability) / 2.0
        base_level = torch.round((self.level_count - 1) * (1.0 - confidence) * complexity_score)
        selected_level = base_level.clamp(0, self.level_count - 1).long()

        override = self._requested_level_tensor(requested_level, pooled.device, pooled.size(0))
        if override is not None:
            selected_level = override

        learned_logits = self.policy(pooled)
        level_axis = torch.arange(self.level_count, device=pooled.device, dtype=pooled.dtype).unsqueeze(0)
        heuristic_prior = -(level_axis - base_level.unsqueeze(-1)).abs()
        logits = learned_logits + heuristic_prior
        return DeliberationDecision(
            logits=logits,
            complexity_score=complexity_score,
            confidence=confidence,
            selected_level=selected_level,
        )

    @staticmethod
    def _pool(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            return hidden.mean(dim=1)
        if hidden.ndim == 2:
            return hidden
        raise ValueError("hidden must have shape [batch, tokens, d_model] or [batch, d_model]")

    def _requested_level_tensor(
        self,
        requested_level: str | int | torch.Tensor | None,
        device: torch.device,
        batch: int,
    ) -> torch.Tensor | None:
        if requested_level is None:
            return None
        if isinstance(requested_level, str):
            parsed = parse_requested_deliberation_level(requested_level)
            if parsed is None and requested_level in DELIBERATION_LEVELS:
                parsed = DELIBERATION_LEVELS.index(requested_level)
            if parsed is None:
                return None
            return torch.full((batch,), parsed, device=device, dtype=torch.long)
        if isinstance(requested_level, int):
            value = max(0, min(self.level_count - 1, requested_level))
            return torch.full((batch,), value, device=device, dtype=torch.long)
        tensor = requested_level.to(device=device, dtype=torch.long).view(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(batch)
        return tensor.clamp(0, self.level_count - 1)


class AmbiguityDetector(nn.Module):
    """Lightweight ambiguity detector using token-distribution entropy."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, hidden: torch.Tensor, *, token_logits: torch.Tensor | None = None) -> torch.Tensor:
        if token_logits is not None:
            probs = torch.softmax(token_logits.float(), dim=-1)
            entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
            normalized = entropy / math.log(max(2, token_logits.size(-1)))
            return normalized.mean(dim=1).clamp(0.0, 1.0).to(hidden.device)
        pooled = AdaptiveDeliberationScheduler._pool(hidden)
        return torch.sigmoid(self.net(pooled)).squeeze(-1)

    @staticmethod
    def should_ask_clarification(ambiguity_score: torch.Tensor, *, threshold: float = 0.60) -> torch.Tensor:
        return ambiguity_score >= threshold


@dataclass
class ThoughtLandmark:
    id: str
    text: str
    importance: float
    source: str
    created_at: str
    detail_ref: str | None = None


class ThoughtLandmarkStore:
    """CPU-side compact reasoning landmarks and long tool-result references."""

    def __init__(self, *, max_landmarks: int = 10, max_landmark_tokens: int = 50) -> None:
        self.max_landmarks = int(max_landmarks)
        self.max_landmark_tokens = int(max_landmark_tokens)
        self._landmarks: list[ThoughtLandmark] = []
        self._tool_results: dict[str, str] = {}

    def add(
        self,
        *,
        text: str,
        importance: float = 0.5,
        source: str = "deliberation",
        detail_ref: str | None = None,
    ) -> ThoughtLandmark:
        landmark = ThoughtLandmark(
            id=uuid.uuid4().hex,
            text=self._truncate(text),
            importance=float(importance),
            source=source,
            created_at=datetime.now(timezone.utc).isoformat(),
            detail_ref=detail_ref,
        )
        self._landmarks.append(landmark)
        self._prune()
        return landmark

    def list_landmarks(self) -> list[ThoughtLandmark]:
        return list(self._landmarks)

    def add_tool_result(self, label: str, content: str) -> str:
        ref = f"tool:{uuid.uuid4().hex}"
        self._tool_results[ref] = content
        self.add(text=f"{label}: see {ref}", importance=0.0, source="tool_result", detail_ref=ref)
        return ref

    def read_tool_result(self, ref: str) -> str:
        return self._tool_results[ref]

    def _truncate(self, text: str) -> str:
        tokens = text.split()
        return " ".join(tokens[: self.max_landmark_tokens])

    def _prune(self) -> None:
        if len(self._landmarks) <= self.max_landmarks:
            return
        indexed = list(enumerate(self._landmarks))
        indexed.sort(key=lambda item: (item[1].importance, item[0]))
        drop = {idx for idx, _ in indexed[: len(self._landmarks) - self.max_landmarks]}
        self._landmarks = [landmark for idx, landmark in enumerate(self._landmarks) if idx not in drop]


class SerialTreeSearchState:
    """CPU state for iterative backtracking without parallel GPU branches."""

    def __init__(self, *, budget_mb: float = 100.0) -> None:
        self.budget_mb = float(budget_mb)
        self.best_plan: str | None = None
        self.best_score: float = -float("inf")
        self.tried_branches: list[tuple[str, float]] = []

    def record(self, plan: str, score: float) -> None:
        score = float(score)
        self.tried_branches.append((plan, score))
        if score > self.best_score:
            self.best_plan = plan
            self.best_score = score
