import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.attention_backend import build_attention_backend
from dopa_coder_n1.model.kv_cache import LayerKVCache, pack_layer_kv
from dopa_coder_n1.model.triton_int4 import triton_int4_status


def test_int4_reference_backend_accepts_packed_cache():
    backend = build_attention_backend("int4_reference")
    q = torch.randn(1, 2, 1, 4)
    k_new = torch.randn(1, 2, 1, 4)
    v_new = torch.randn(1, 2, 1, 4)
    cache = LayerKVCache(k=torch.randn(1, 2, 3, 4), v=torch.randn(1, 2, 3, 4))
    packed = pack_layer_kv(cache)
    y, new_cache = backend.attention(
        q,
        k_new,
        v_new,
        kv_cache=packed,
        n_heads=2,
        n_kv_heads=2,
        dropout_p=0.0,
        is_training=False,
    )
    assert y.shape == q.shape
    assert new_cache[0].shape == (1, 2, 4, 4)
    assert new_cache[1].shape == (1, 2, 4, 4)


def test_triton_int4_backend_accepts_packed_cache_with_fallback():
    backend = build_attention_backend("triton_int4")
    q = torch.randn(1, 2, 1, 4)
    k_new = torch.randn(1, 2, 1, 4)
    v_new = torch.randn(1, 2, 1, 4)
    cache = LayerKVCache(k=torch.randn(1, 2, 3, 4), v=torch.randn(1, 2, 3, 4))
    packed = pack_layer_kv(cache)
    y, new_cache = backend.attention(
        q,
        k_new,
        v_new,
        kv_cache=packed,
        n_heads=2,
        n_kv_heads=2,
        dropout_p=0.0,
        is_training=False,
    )
    assert y.shape == q.shape
    assert new_cache[0].shape == (1, 2, 4, 4)
    assert new_cache[1].shape == (1, 2, 4, 4)
    assert "usable" in triton_int4_status(q.device)


def test_tiny_unit_uses_triton_int4_backend():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    assert cfg.model.attention_backend == "triton_int4"
