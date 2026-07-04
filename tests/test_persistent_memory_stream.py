import tempfile
from pathlib import Path

from dopa_coder_n1.model.persistent_memory import PersistentMemoryStore


def test_persistent_memory_write_threshold_and_injection_format():
    with tempfile.TemporaryDirectory() as tmp:
        store = PersistentMemoryStore(Path(tmp) / "memory.sqlite", max_bytes=10_000)

        rejected = store.write_memory(memory_type="fact", content="scratch note", importance=0.20)
        accepted = store.write_memory(
            memory_type="preference",
            content="User keeps complete DoAP projects on D drive and wants strict paper alignment.",
            importance=0.82,
        )

        assert not rejected.accepted
        assert accepted.accepted

        result = store.search("where should DoAP projects live", limit=1)

        assert len(result.memories) == 1
        assert result.memories[0].type == "preference"
        assert "<|memory|>" in result.injection_prefix
        assert "D drive" in result.injection_prefix


def test_persistent_memory_prune_keeps_high_value_record_under_budget():
    with tempfile.TemporaryDirectory() as tmp:
        store = PersistentMemoryStore(Path(tmp) / "memory.sqlite", max_bytes=240)
        low = store.write_memory(memory_type="fact", content="low value " * 16, importance=0.55)
        high = store.write_memory(memory_type="fact", content="critical architecture " * 16, importance=0.95)

        assert low.accepted and high.accepted
        report = store.prune_to_budget(max_bytes=150)
        remaining = store.search("critical architecture", limit=5).memories

        assert report.deleted_count >= 1
        assert any(memory.id == high.memory.id for memory in remaining)
        assert not any(memory.id == low.memory.id for memory in remaining)


def test_persistent_memory_default_budget_is_10gb():
    with tempfile.TemporaryDirectory() as tmp:
        store = PersistentMemoryStore(Path(tmp) / "memory.sqlite")
        assert store.max_bytes == 10 * 1024**3
