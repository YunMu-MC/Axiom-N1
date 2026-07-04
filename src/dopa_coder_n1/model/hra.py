from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from dopa_coder_n1.model.rust_core import RustRetrievalBackend


MB = 1024 * 1024


@dataclass(slots=True)
class HRAConfig:
    max_context_tokens: int = 258_000
    chunk_tokens: int = 512
    hot_window_tokens: int = 4096
    level2_span_tokens: int = 4096
    top_k_chunks: int = 4
    budget_d_model: int = 8192

    def __post_init__(self) -> None:
        for name in (
            "max_context_tokens",
            "chunk_tokens",
            "hot_window_tokens",
            "level2_span_tokens",
            "top_k_chunks",
            "budget_d_model",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(slots=True)
class HRAIngestReport:
    session_id: str
    indexed_tokens: int
    chunk_count: int
    level1_summary_count: int
    level2_summary_count: int
    hot_window_tokens: int
    max_context_tokens: int
    level1_cpu_mb: float
    level2_gpu_mb: float
    transient_kv_mb_top_k: float


@dataclass(slots=True)
class HRAChunk:
    chunk_id: int
    token_start: int
    token_end: int
    vector: torch.Tensor
    text: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "vector": self.vector.cpu(),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "HRAChunk":
        return cls(
            chunk_id=int(raw["chunk_id"]),
            token_start=int(raw["token_start"]),
            token_end=int(raw["token_end"]),
            vector=raw["vector"].detach().cpu(),
            text=str(raw.get("text", "")),
        )


@dataclass(slots=True)
class HRARetrievedChunk:
    chunk_id: int
    token_start: int
    token_end: int
    text: str
    score: float

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "text": self.text,
            "score": self.score,
        }


@dataclass(slots=True)
class HRARetrievalResult:
    session_id: str
    loaded_chunk_count: int
    chunks: list[HRARetrievedChunk]
    transient_kv_mb: float
    per_chunk_kv_mb: float
    release_after_attention: bool
    backend: str


class HierarchicalRetrievalAttention:
    """SSD-friendly HRA manager for 258K context retrieval.

    Tensor pooling remains in PyTorch because it is directly produced by the
    Hot Core. Text ranking for chunk retrieval is delegated to the Rust backend
    when query text is available.
    """

    def __init__(
        self,
        config: HRAConfig | None = None,
        *,
        storage_dir: Path | None = None,
        rust_backend: RustRetrievalBackend | None = None,
    ):
        self.config = config or HRAConfig()
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None
        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.rust_backend = rust_backend or RustRetrievalBackend.default()
        self._sessions: dict[str, list[HRAChunk]] = {}
        self._level2_counts: dict[str, int] = {}

    def ingest(
        self,
        session_id: str,
        hidden: torch.Tensor,
        texts: Iterable[str] | None = None,
    ) -> HRAIngestReport:
        if hidden.ndim != 2:
            raise ValueError("hidden must be [tokens, d_model]")
        cfg = self.config
        hidden = hidden[: cfg.max_context_tokens].detach().cpu().float()
        text_list = list(texts) if texts is not None else ["" for _ in range(hidden.shape[0])]
        if len(text_list) < hidden.shape[0]:
            text_list.extend([""] * (hidden.shape[0] - len(text_list)))
        chunks: list[HRAChunk] = []
        for chunk_id, start in enumerate(range(0, hidden.shape[0], cfg.chunk_tokens)):
            end = min(start + cfg.chunk_tokens, hidden.shape[0])
            vector = hidden[start:end].mean(dim=0)
            text = " ".join(str(item) for item in text_list[start:end]).strip()
            chunks.append(HRAChunk(chunk_id, start, end, vector, text))
        level2_ids = {chunk.token_start // cfg.level2_span_tokens for chunk in chunks}
        self._sessions[session_id] = chunks
        self._level2_counts[session_id] = len(level2_ids)
        self._persist(session_id, chunks, len(level2_ids))
        return HRAIngestReport(
            session_id=session_id,
            indexed_tokens=int(hidden.shape[0]),
            chunk_count=len(chunks),
            level1_summary_count=len(chunks),
            level2_summary_count=len(level2_ids),
            hot_window_tokens=cfg.hot_window_tokens,
            max_context_tokens=cfg.max_context_tokens,
            level1_cpu_mb=round(len(chunks) * cfg.budget_d_model * 2 / MB, 4),
            level2_gpu_mb=round(len(level2_ids) * cfg.budget_d_model * 2 / MB, 4),
            transient_kv_mb_top_k=round(self._kv_mb(cfg.top_k_chunks), 4),
        )

    def retrieve(
        self,
        session_id: str,
        query_vector: torch.Tensor,
        *,
        top_k: int | None = None,
        query_text: str | None = None,
    ) -> HRARetrievalResult:
        chunks = self._load(session_id)
        limit = max(1, int(top_k or self.config.top_k_chunks))
        backend = "python"
        selected: list[tuple[float, HRAChunk]]
        if query_text and self.rust_backend.available():
            ranked = self.rust_backend.rank_texts(
                query=query_text,
                rows=[{"id": str(chunk.chunk_id), "text": chunk.text} for chunk in chunks],
                limit=limit,
            )
            by_id = {chunk.chunk_id: chunk for chunk in chunks}
            selected = [(float(item["score"]), by_id[int(item["id"])]) for item in ranked if int(item["id"]) in by_id]
            backend = "rust"
        else:
            q = query_vector.detach().cpu().float().flatten()
            selected = [(self._cosine(q, chunk.vector), chunk) for chunk in chunks]
            selected.sort(key=lambda item: item[0], reverse=True)
            selected = selected[:limit]
        retrieved = [
            HRARetrievedChunk(
                chunk_id=chunk.chunk_id,
                token_start=chunk.token_start,
                token_end=chunk.token_end,
                text=chunk.text,
                score=round(float(score), 4),
            )
            for score, chunk in selected[:limit]
        ]
        return HRARetrievalResult(
            session_id=session_id,
            loaded_chunk_count=len(retrieved),
            chunks=retrieved,
            transient_kv_mb=round(self._kv_mb(len(retrieved)), 4),
            per_chunk_kv_mb=round(self._kv_mb(1), 4),
            release_after_attention=True,
            backend=backend,
        )

    def _persist(self, session_id: str, chunks: list[HRAChunk], level2_count: int) -> None:
        if self.storage_dir is None:
            return
        torch.save(
            {
                "chunks": [chunk.to_dict() for chunk in chunks],
                "level2_count": level2_count,
            },
            self.storage_dir / f"{session_id}.hra.pt",
        )

    def _load(self, session_id: str) -> list[HRAChunk]:
        if session_id in self._sessions:
            return self._sessions[session_id]
        if self.storage_dir is None:
            raise KeyError(f"unknown HRA session: {session_id}")
        path = self.storage_dir / f"{session_id}.hra.pt"
        if not path.exists():
            raise KeyError(f"unknown HRA session: {session_id}")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        chunks = [HRAChunk.from_dict(item) for item in raw["chunks"]]
        self._sessions[session_id] = chunks
        self._level2_counts[session_id] = int(raw.get("level2_count", 0))
        return chunks

    def _kv_mb(self, chunk_count: int) -> float:
        cfg = self.config
        return chunk_count * cfg.chunk_tokens * cfg.budget_d_model * 2 / MB

    @staticmethod
    def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        dim = min(left.numel(), right.numel())
        if dim == 0:
            return 0.0
        a = left[:dim]
        b = right[:dim]
        denom = a.norm().clamp_min(1e-8) * b.norm().clamp_min(1e-8)
        return float(torch.dot(a, b) / denom)
