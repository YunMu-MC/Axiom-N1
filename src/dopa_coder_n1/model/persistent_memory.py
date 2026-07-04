from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any
from uuid import uuid4

from dopa_coder_n1.model.rust_core import RustRetrievalBackend


@dataclass(slots=True)
class MemoryRecord:
    id: str
    type: str
    content: str
    timestamp: float
    importance: float
    access_count: int
    last_access: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class MemoryWriteResult:
    accepted: bool
    memory: MemoryRecord | None = None
    reason: str = ""
    payload_bytes: int = 0


@dataclass(slots=True)
class MemorySearchResult:
    query: str
    memories: list[MemoryRecord]
    scores: list[float]
    injection_prefix: str
    backend: str


@dataclass(slots=True)
class MemoryPruneReport:
    budget_bytes: int
    remaining_payload_bytes: int
    deleted_count: int
    deleted_ids: list[str]


class PersistentMemoryStore:
    """DoAP persistent memory stream with a 10GB default disk budget."""

    def __init__(
        self,
        db_path: Path,
        *,
        max_bytes: int = 10 * 1024**3,
        importance_threshold: float = 0.50,
        rust_backend: RustRetrievalBackend | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.importance_threshold = float(importance_threshold)
        self.rust_backend = rust_backend or RustRetrievalBackend.default()
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    last_access REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    payload_bytes INTEGER NOT NULL
                )
                """
            )

    def write_memory(
        self,
        *,
        memory_type: str,
        content: str,
        importance: float,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        importance = float(importance)
        if importance <= self.importance_threshold:
            return MemoryWriteResult(False, reason="importance_below_threshold")
        now = time.time()
        metadata = dict(metadata or {})
        memory = MemoryRecord(
            id=str(uuid4()),
            type=memory_type,
            content=content,
            timestamp=now,
            importance=importance,
            access_count=0,
            last_access=now,
            metadata=metadata,
        )
        payload_bytes = len(content.encode("utf-8")) + len(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories(
                    id, type, content, timestamp, importance,
                    access_count, last_access, metadata, payload_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.type,
                    memory.content,
                    memory.timestamp,
                    memory.importance,
                    memory.access_count,
                    memory.last_access,
                    json.dumps(memory.metadata, ensure_ascii=False, separators=(",", ":")),
                    payload_bytes,
                ),
            )
        return MemoryWriteResult(True, memory=memory, payload_bytes=payload_bytes)

    def search(self, query: str, limit: int = 5) -> MemorySearchResult:
        rows = self._rows()
        limit = max(0, int(limit))
        backend = "python"
        if self.rust_backend.available():
            ranked = self.rust_backend.rank_texts(
                query=query,
                rows=[{"id": row["id"], "text": row["content"]} for row in rows],
                limit=limit,
            )
            score_by_id = {item["id"]: float(item["score"]) for item in ranked}
            order = [item["id"] for item in ranked]
            rows_by_id = {row["id"]: row for row in rows}
            selected_rows = [rows_by_id[item] for item in order if item in rows_by_id]
            scores = [score_by_id[row["id"]] for row in selected_rows]
            backend = "rust"
        else:
            scored = [(self._lexical_score(query, row["content"]), row) for row in rows]
            scored.sort(key=lambda item: item[0], reverse=True)
            selected_rows = [row for _, row in scored[:limit]]
            scores = [score for score, _ in scored[:limit]]
        memories = [self._record_from_row(row, access_increment=1) for row in selected_rows]
        self._mark_accessed([memory.id for memory in memories])
        injection = "\n".join(f"<|memory|> {memory.content} <|/memory|>" for memory in memories)
        return MemorySearchResult(
            query=query,
            memories=memories,
            scores=[round(float(score), 4) for score in scores],
            injection_prefix=injection,
            backend=backend,
        )

    def prune_to_budget(
        self,
        *,
        max_bytes: int | None = None,
        gamma: float = 0.99,
        tau_seconds: float = 30 * 24 * 3600,
        access_weight: float = 0.01,
    ) -> MemoryPruneReport:
        budget = int(max_bytes or self.max_bytes)
        now = time.time()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, importance, access_count, last_access, payload_bytes FROM memories"
            ).fetchall()
            total = sum(int(row["payload_bytes"]) for row in rows)
            ranked = []
            for row in rows:
                delta = max(0.0, now - float(row["last_access"]))
                score = float(row["importance"]) * (gamma ** (delta / tau_seconds))
                score += access_weight * int(row["access_count"])
                ranked.append((score, row))
            ranked.sort(key=lambda item: item[0])
            deleted: list[str] = []
            while total > budget and len(ranked) - len(deleted) > 1:
                _, row = ranked[len(deleted)]
                deleted.append(str(row["id"]))
                total -= int(row["payload_bytes"])
            if deleted:
                conn.executemany("DELETE FROM memories WHERE id = ?", [(item,) for item in deleted])
        return MemoryPruneReport(
            budget_bytes=budget,
            remaining_payload_bytes=max(0, total),
            deleted_count=len(deleted),
            deleted_ids=deleted,
        )

    def _rows(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, type, content, timestamp, importance, access_count, last_access, metadata
                FROM memories
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def _mark_accessed(self, ids: list[str]) -> None:
        if not ids:
            return
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_access = ?
                WHERE id = ?
                """,
                [(now, item) for item in ids],
            )

    def _record_from_row(self, row: dict[str, Any], *, access_increment: int = 0) -> MemoryRecord:
        return MemoryRecord(
            id=str(row["id"]),
            type=str(row["type"]),
            content=str(row["content"]),
            timestamp=float(row["timestamp"]),
            importance=float(row["importance"]),
            access_count=int(row["access_count"]) + access_increment,
            last_access=float(row["last_access"]),
            metadata=json.loads(str(row["metadata"] or "{}")),
        )

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        q = {item.lower() for item in query.split() if item}
        t = {item.lower().strip(".,:;()[]{}") for item in text.split() if item}
        if not q:
            return 0.0
        return len(q & t) / len(q)
