import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.model.hra import HRAConfig, HierarchicalRetrievalAttention
from dopa_coder_n1.model.rust_core import RustRetrievalBackend


def test_hra_ingest_builds_258k_paper_budget_summaries():
    cfg = HRAConfig(
        max_context_tokens=258_000,
        chunk_tokens=8,
        hot_window_tokens=16,
        level2_span_tokens=32,
        top_k_chunks=4,
        budget_d_model=8192,
    )
    hra = HierarchicalRetrievalAttention(cfg)
    hidden = torch.randn(34, 16)

    report = hra.ingest("s1", hidden)

    assert report.chunk_count == 5
    assert report.level1_summary_count == 5
    assert report.level2_summary_count == 2
    assert report.hot_window_tokens == 16
    assert report.max_context_tokens == 258_000
    assert report.level2_gpu_mb < 1.0


def test_hra_retrieves_only_top_k_transient_chunks_and_estimates_kv():
    cfg = HRAConfig(chunk_tokens=4, hot_window_tokens=8, level2_span_tokens=16, top_k_chunks=2, budget_d_model=8192)
    hra = HierarchicalRetrievalAttention(cfg)
    hidden = torch.zeros(16, 8)
    hidden[4:8, 0] = 4.0
    hidden[8:12, 1] = 4.0
    hidden[12:16, 2] = 4.0
    hra.ingest("s2", hidden, texts=["alpha"] * 4 + ["database"] * 4 + ["triton"] * 4 + ["memory"] * 4)

    result = hra.retrieve(
        "s2",
        torch.tensor([0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        top_k=2,
        query_text="triton int4 attention",
    )

    assert result.loaded_chunk_count == 2
    assert result.chunks[0].token_start == 8
    assert "triton" in result.chunks[0].text
    assert result.backend == "rust"
    assert result.transient_kv_mb < 50.0
    assert result.release_after_attention


def test_hra_can_persist_chunk_index_to_ssd_path():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = HRAConfig(chunk_tokens=4, hot_window_tokens=4, level2_span_tokens=8)
        hra = HierarchicalRetrievalAttention(cfg, storage_dir=Path(tmp))
        hra.ingest("persisted", torch.randn(9, 4))

        loaded = HierarchicalRetrievalAttention(cfg, storage_dir=Path(tmp))
        result = loaded.retrieve("persisted", torch.randn(4), top_k=1)

        assert result.loaded_chunk_count == 1
        assert (Path(tmp) / "persisted.hra.pt").exists()


def test_rust_backend_is_available_for_text_ranking():
    backend = RustRetrievalBackend.default()
    assert backend.available(), backend.binary
    ranked = backend.rank_texts(
        query="triton int4 attention",
        rows=[
            {"id": "a", "text": "database migration rollback"},
            {"id": "b", "text": "triton int4 attention cache"},
        ],
        limit=1,
    )
    assert ranked[0]["id"] == "b"
