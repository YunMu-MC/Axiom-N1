import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.model.cognitive import CognitiveSearchController
from dopa_coder_n1.model.hra import HRAConfig, HierarchicalRetrievalAttention
from dopa_coder_n1.model.persistent_memory import PersistentMemoryStore


def test_cognitive_controller_fuses_web_memory_and_long_context_when_curious():
    with tempfile.TemporaryDirectory() as tmp:
        memory = PersistentMemoryStore(Path(tmp) / "memory.sqlite")
        memory.write_memory(
            memory_type="summary",
            content="The project requires 258K HRA and a 10GB persistent memory stream.",
            importance=0.9,
        )
        hra = HierarchicalRetrievalAttention(HRAConfig(chunk_tokens=4, hot_window_tokens=4, level2_span_tokens=8))
        hidden = torch.zeros(8, 4)
        hidden[4:8, 0] = 3.0
        hra.ingest("s1", hidden, texts=["boring"] * 4 + ["hra"] * 4)

        controller = CognitiveSearchController(
            threshold=0.20,
            search_fn=lambda query: ["web snippet about DoAP fleeting learning"],
            memory_store=memory,
            long_context=hra,
        )

        result = controller.maybe_search(
            "DoAP HRA memory",
            confidence=0.10,
            session_id="s1",
            query_vector=torch.tensor([4.0, 0.0, 0.0, 0.0]),
        )

        assert result is not None
        assert any("web snippet" in item for item in result.snippets)
        assert any("258K HRA" in item for item in result.memory_snippets)
        assert result.long_context_chunks[0]["token_start"] == 4
        assert result.skeleton["kind"] == "induced_skeleton"
