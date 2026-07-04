from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
from torch import nn


@dataclass
class CognitiveSearchResult:
    query: str
    snippets: list[str]
    skeleton: dict
    confidence: float
    memory_snippets: list[str] = field(default_factory=list)
    long_context_chunks: list[dict[str, Any]] = field(default_factory=list)


class InductiveSkeletonGenerator:
    """Small deterministic ISG interface for few-example method abstraction."""

    def induce(self, solved_instances: Iterable[dict]) -> dict:
        names: list[str] = []
        ops: list[str] = []
        for item in solved_instances:
            if name := item.get("name"):
                names.append(str(name))
            skeleton = item.get("skeleton") or {}
            for step in skeleton.get("steps", []):
                op = step.get("op")
                if op is not None:
                    ops.append(str(op))
        unique_ops = list(dict.fromkeys(ops))
        return {
            "name": "_".join(names[:3]) or "induced_method",
            "kind": "induced_skeleton",
            "steps": [{"op": op} for op in unique_ops] or [{"op": "analyze"}, {"op": "solve"}],
        }


class KnowledgeDistiller(nn.Module):
    """Distills retrieved snippets into a temporary memory vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.Tanh(),
        )

    def forward(self, snippets: list[str], *, device: torch.device) -> torch.Tensor:
        if not snippets:
            stats = torch.zeros(4, device=device)
        else:
            lengths = torch.tensor([len(x) for x in snippets], device=device, dtype=torch.float32)
            code_marks = torch.tensor(
                [x.count("def ") + x.count("class ") for x in snippets],
                device=device,
                dtype=torch.float32,
            )
            stats = torch.stack(
                [
                    lengths.mean(),
                    lengths.max(),
                    code_marks.mean(),
                    torch.tensor(float(len(snippets)), device=device),
                ]
            )
            stats = stats / stats.abs().max().clamp_min(1.0)
        return self.proj(stats).unsqueeze(0)


class CognitiveSearchController:
    """Curiosity-triggered search/distillation boundary.

    Network search, persistent memory, and HRA long-context retrieval are
    injected by the caller. This keeps training deterministic while matching
    the paper's query -> snippets/memory/chunks -> temporary skeleton -> fast
    weight lifecycle.
    """

    def __init__(
        self,
        *,
        threshold: float,
        search_fn: Callable[[str], list[str]] | None = None,
        memory_store: Any | None = None,
        long_context: Any | None = None,
    ):
        self.threshold = threshold
        self.search_fn = search_fn
        self.memory_store = memory_store
        self.long_context = long_context
        self.isg = InductiveSkeletonGenerator()

    def maybe_search(
        self,
        query: str,
        confidence: float,
        *,
        session_id: str | None = None,
        query_vector: torch.Tensor | None = None,
        memory_limit: int = 5,
        long_context_top_k: int = 4,
    ) -> CognitiveSearchResult | None:
        if confidence >= self.threshold:
            return None
        snippets = self.search_fn(query) if self.search_fn is not None else []
        memory_snippets: list[str] = []
        if self.memory_store is not None:
            memory_result = self.memory_store.search(query, limit=memory_limit)
            memory_snippets = [memory.content for memory in memory_result.memories]
        long_context_chunks: list[dict[str, Any]] = []
        if self.long_context is not None and session_id is not None and query_vector is not None:
            chunk_result = self.long_context.retrieve(
                session_id,
                query_vector,
                top_k=long_context_top_k,
                query_text=query,
            )
            long_context_chunks = [chunk.as_dict() for chunk in chunk_result.chunks]
        solved = []
        for item in snippets[:3]:
            solved.append({"name": query, "skeleton": {"steps": [{"op": "retrieve_web", "text": item}]}})
        for item in memory_snippets[:3]:
            solved.append({"name": query, "skeleton": {"steps": [{"op": "retrieve_memory", "text": item}]}})
        for item in long_context_chunks[:3]:
            solved.append({"name": query, "skeleton": {"steps": [{"op": "retrieve_context", "text": item["text"]}]}})
        skeleton = self.isg.induce(solved or [{"name": query, "skeleton": {"steps": [{"op": "retrieve"}]}}])
        return CognitiveSearchResult(
            query=query,
            snippets=snippets,
            skeleton=skeleton,
            confidence=confidence,
            memory_snippets=memory_snippets,
            long_context_chunks=long_context_chunks,
        )
