from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen


class CuriositySearchClient:
    """Lightweight HTTP client for the standalone DoAP Curiosity Search service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765", timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def probe(
        self,
        *,
        task: str,
        current_answer_draft: str = "",
        model_confidence: float = 1.0,
        difficulty_score: float = 0.0,
        local_context_insufficient: bool = False,
        answer_contains_time_sensitive_claim: bool = False,
        sources_conflict: bool = False,
    ) -> dict[str, Any]:
        return self._post(
            "/curiosity/probe",
            {
                "task": task,
                "current_answer_draft": current_answer_draft,
                "model_confidence": model_confidence,
                "difficulty_score": difficulty_score,
                "local_context_insufficient": local_context_insufficient,
                "answer_contains_time_sensitive_claim": answer_contains_time_sensitive_claim,
                "sources_conflict": sources_conflict,
            },
        )

    def search(
        self,
        *,
        session_id: str,
        query_plan: list[str],
        source_policy: dict[str, Any] | None = None,
        time_policy: dict[str, Any] | None = None,
        max_sources_per_query: int = 5,
    ) -> dict[str, Any]:
        return self._post(
            "/search",
            {
                "session_id": session_id,
                "query_plan": query_plan,
                "source_policy": source_policy or {},
                "time_policy": time_policy or {},
                "max_sources_per_query": max_sources_per_query,
            },
        )

    def distill(
        self,
        *,
        session_id: str | None = None,
        evidence_cards: list[dict[str, Any]] | None = None,
        task_context: str = "",
    ) -> dict[str, Any]:
        return self._post(
            "/distill",
            {
                "session_id": session_id,
                "evidence_cards": evidence_cards,
                "task_context": task_context,
            },
        )

    def forget(self, *, session_id: str) -> dict[str, Any]:
        return self._post("/forget", {"session_id": session_id})

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
