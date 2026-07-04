import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.model.kv_cache import (
    ColdSelectiveKVCache,
    ColdSelectiveKVState,
    HotKVCache,
    LayerKVCache,
    dequantize_int4_tensor,
    pack_layer_kv,
    quantize_int4_tensor,
    unpack_layer_kv,
)


def test_int4_pack_roundtrip_shape():
    x = torch.randn(7, 5)
    packed, scale = quantize_int4_tensor(x)
    y = dequantize_int4_tensor(packed, scale, tuple(x.shape))
    assert y.shape == x.shape


def test_layer_kv_pack_roundtrip_shape_and_device():
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    cache = LayerKVCache(
        k=k,
        v=v,
        k_summary=torch.randn(1, 2, 1, 4),
        v_summary=torch.randn(1, 2, 1, 4),
        has_summary=True,
    )
    packed = pack_layer_kv(cache)
    restored = unpack_layer_kv(packed, device=k.device, dtype=k.dtype)
    assert restored.k.shape == k.shape
    assert restored.v.shape == v.shape
    assert restored.k.device == k.device
    assert restored.v.dtype == v.dtype
    assert restored.has_summary
    attn_k, attn_v = restored.attention_kv()
    assert attn_k.shape[2] == k.shape[2] + 1
    assert attn_v.shape[2] == v.shape[2] + 1


def test_hot_kv_cache_sliding_summary():
    cache = HotKVCache(layers=1, window=2, d_model=4, decay=0.5)
    cache.append(0, torch.ones(4), torch.ones(4) * 2)
    cache.append(0, torch.ones(4) * 3, torch.ones(4) * 4)
    cache.append(0, torch.ones(4) * 5, torch.ones(4) * 6)
    k_sum, v_sum, k, v = cache.get_dense(0)
    assert k.shape == (2, 4)
    assert torch.allclose(k_sum, torch.ones(4) * 0.5)
    assert torch.allclose(v_sum, torch.ones(4))
    packed = cache.packed_window(0)
    assert packed.shape == (2, 4)


def test_cold_selective_cache_only_hot_units():
    cache = ColdSelectiveKVCache(max_units=1, window=2, head_dim=4)
    cache.append("a", torch.ones(4), torch.ones(4))
    assert cache.packed("a") is None
    cache.mark_hot("a")
    cache.append("a", torch.ones(4), torch.ones(4))
    assert cache.packed("a") is not None
    cache.mark_hot("b")
    assert cache.packed("a") is None


def test_cold_selective_state_packs_window():
    state = ColdSelectiveKVState(max_units=1, window=2)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    state.put("unit_a", LayerKVCache(k=k, v=v))
    restored = state.get("unit_a", device=k.device, dtype=k.dtype)
    assert restored is not None
    assert restored.k.shape == (1, 1, 2, 4)
    state.put("unit_b", LayerKVCache(k=k, v=v))
    assert state.get("unit_a") is None
    assert state.get("unit_b") is not None
