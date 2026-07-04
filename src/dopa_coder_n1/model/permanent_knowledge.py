from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KN_CREATE = "[KN_CREATE]"
KN_UPDATE = "[KN_UPDATE]"
KN_MERGE = "[KN_MERGE]"
KN_DELETE = "[KN_DELETE]"
KN_QUERY = "[KN_QUERY]"
KNOWLEDGE_OPERATIONS = (KN_CREATE, KN_UPDATE, KN_MERGE, KN_DELETE, KN_QUERY)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_+#]+", text.lower())


def _stable_embedding(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % dim
        sign = 1.0 if (value >> 8) & 1 else -1.0
        vec[index] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine(left: list[float], right: list[float]) -> float:
    denom = math.sqrt(sum(v * v for v in left)) * math.sqrt(sum(v * v for v in right))
    if denom == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denom


@dataclass
class KnowledgePoint:
    id: str
    rule: str
    domain: str
    embedding: list[float]
    importance: float
    access_count: int
    last_access: str
    created_from: str
    dependencies: list[str] = field(default_factory=list)
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KnowledgePoint":
        return cls(
            id=str(raw["id"]),
            rule=str(raw["rule"]),
            domain=str(raw.get("domain", "general")),
            embedding=[float(x) for x in raw.get("embedding", [])][:128],
            importance=float(raw.get("importance", 0.5)),
            access_count=int(raw.get("access_count", 0)),
            last_access=str(raw.get("last_access", _now())),
            created_from=str(raw.get("created_from", "unknown")),
            dependencies=[str(x) for x in raw.get("dependencies", [])],
            positive_examples=[str(x) for x in raw.get("positive_examples", [])],
            negative_examples=[str(x) for x in raw.get("negative_examples", [])],
        )


class PermanentKnowledgeBase:
    """Small JSON-file knowledge base with deterministic 128-dim embeddings."""

    def __init__(
        self,
        root: str | Path = "knowledge_base",
        *,
        max_files: int = 500,
        max_file_bytes: int = 20 * 1024,
        embedding_dim: int = 128,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_files = int(max_files)
        self.max_file_bytes = int(max_file_bytes)
        self.embedding_dim = int(embedding_dim)

    def create(
        self,
        *,
        rule: str,
        domain: str,
        importance: float = 0.5,
        created_from: str,
        dependencies: list[str] | None = None,
    ) -> KnowledgePoint:
        point = KnowledgePoint(
            id=uuid.uuid4().hex,
            rule=rule.strip(),
            domain=domain.strip() or "general",
            embedding=_stable_embedding(f"{domain} {rule}", self.embedding_dim),
            importance=float(importance),
            access_count=0,
            last_access=_now(),
            created_from=created_from,
            dependencies=list(dependencies or []),
        )
        self._save(point)
        return point

    def query(self, text: str, *, top_k: int = 3) -> list[KnowledgePoint]:
        query_embedding = _stable_embedding(text, self.embedding_dim)
        query_tokens = set(_tokens(text))
        scored: list[tuple[float, KnowledgePoint]] = []
        for point in self.list_points():
            point_text = f"{point.domain} {point.rule} {' '.join(point.positive_examples)}"
            point_tokens = set(_tokens(point_text))
            overlap = len(query_tokens & point_tokens) / max(1, len(query_tokens))
            vector_score = _cosine(query_embedding, point.embedding)
            score = 0.70 * overlap + 0.25 * vector_score + 0.05 * point.importance
            scored.append((score, point))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = [point for _, point in scored[: max(0, top_k)]]
        for point in results:
            point.access_count += 1
            point.last_access = _now()
            self._save(point)
        return results

    def update(
        self,
        point_id: str,
        *,
        rule: str | None = None,
        domain: str | None = None,
        importance: float | None = None,
        add_positive: str | None = None,
        add_negative: str | None = None,
    ) -> KnowledgePoint:
        point = self._load(point_id)
        if rule is not None:
            point.rule = rule.strip()
        if domain is not None:
            point.domain = domain.strip() or point.domain
        if importance is not None:
            point.importance = float(importance)
        if add_positive:
            point.positive_examples.append(add_positive.strip())
        if add_negative:
            point.negative_examples.append(add_negative.strip())
        point.embedding = _stable_embedding(
            f"{point.domain} {point.rule} {' '.join(point.positive_examples)}",
            self.embedding_dim,
        )
        self._save(point)
        return point

    def merge(self, primary_id: str, absorbed_id: str) -> KnowledgePoint:
        primary = self._load(primary_id)
        absorbed = self._load(absorbed_id)
        if absorbed.id not in primary.dependencies:
            primary.dependencies.append(absorbed.id)
        for dependency in absorbed.dependencies:
            if dependency not in primary.dependencies:
                primary.dependencies.append(dependency)
        primary.positive_examples.extend(absorbed.positive_examples)
        primary.negative_examples.extend(absorbed.negative_examples)
        if absorbed.rule and absorbed.rule not in primary.rule:
            primary.rule = f"{primary.rule} Related rule: {absorbed.rule}"
        primary.importance = max(primary.importance, absorbed.importance)
        primary.embedding = _stable_embedding(
            f"{primary.domain} {primary.rule} {' '.join(primary.positive_examples)}",
            self.embedding_dim,
        )
        self._save(primary)
        self._path(absorbed_id).unlink(missing_ok=True)
        return primary

    def delete(self, point_id: str) -> bool:
        path = self._path(point_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_points(self) -> list[KnowledgePoint]:
        points = []
        for path in sorted(self.root.glob("*.json")):
            try:
                points.append(KnowledgePoint.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return points

    def prune_to_limit(self) -> dict[str, int | list[str]]:
        points = self.list_points()
        if len(points) <= self.max_files:
            return {"deleted_count": 0, "deleted_ids": []}
        points.sort(key=lambda point: (point.importance, point.access_count, point.last_access))
        delete_count = len(points) - self.max_files
        deleted = []
        for point in points[:delete_count]:
            if self.delete(point.id):
                deleted.append(point.id)
        return {"deleted_count": len(deleted), "deleted_ids": deleted}

    def _path(self, point_id: str) -> Path:
        return self.root / f"{point_id}.json"

    def _load(self, point_id: str) -> KnowledgePoint:
        path = self._path(point_id)
        if not path.exists():
            raise KeyError(f"knowledge point not found: {point_id}")
        return KnowledgePoint.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _save(self, point: KnowledgePoint) -> None:
        payload = asdict(point)
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        if len(text.encode("utf-8")) > self.max_file_bytes:
            point.positive_examples = point.positive_examples[-8:]
            point.negative_examples = point.negative_examples[-8:]
            point.rule = point.rule[: min(len(point.rule), 4096)]
            payload = asdict(point)
            text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        if len(text.encode("utf-8")) > self.max_file_bytes:
            raise ValueError(f"knowledge point exceeds {self.max_file_bytes} bytes")
        self._path(point.id).write_text(text, encoding="utf-8")


class FailureDrivenKnowledgeLearner:
    """Convert an overconfident execution failure into a durable rule point."""

    def __init__(self, knowledge_base: PermanentKnowledgeBase) -> None:
        self.knowledge_base = knowledge_base

    def learn_from_failure(
        self,
        *,
        user_requirement: str,
        failed_output: str,
        corrected_output: str,
    ) -> KnowledgePoint:
        rule = (
            "When a similar requirement appears, prefer the corrected pattern: "
            f"{corrected_output.strip()}. Avoid the failed pattern: {failed_output.strip()}."
        )
        domain = self._infer_domain(user_requirement, corrected_output)
        point = self.knowledge_base.create(
            rule=rule,
            domain=domain,
            importance=0.85,
            created_from="user_feedback",
        )
        self.knowledge_base.update(
            point.id,
            add_positive=corrected_output,
            add_negative=failed_output,
        )
        return self.knowledge_base._load(point.id)

    @staticmethod
    def _infer_domain(user_requirement: str, corrected_output: str) -> str:
        text = f"{user_requirement} {corrected_output}".lower()
        if "c++" in text or "std::" in text or "vector" in text:
            return "C++"
        if "python" in text:
            return "Python"
        if "rust" in text:
            return "Rust"
        return "general"
