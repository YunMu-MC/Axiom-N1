import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.data.skeleton_tokenizer import SkeletonTokenizer
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.kv_cache import PackedLayerKV
from dopa_coder_n1.model.skeleton import SkeletonBatch


def test_tiny_forward_shapes():
    cfg = DOPAConfig.from_yaml("configs/tiny.yaml")
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    skel_tok = SkeletonTokenizer(cfg.dopa.skeleton_vocab_size)
    skel = SkeletonBatch(
        token_ids=torch.tensor(
            [
                skel_tok.encode({"name": "grid_bfs", "steps": [{"op": "graph_search"}]}, max_len=32),
                skel_tok.encode({"name": "interval_sort", "steps": [{"op": "sort"}]}, max_len=32),
            ]
        )
    )
    out = model(x, labels=x, skeleton=skel, return_aux=True)
    assert out.logits.shape == (2, 16, cfg.model.vocab_size)
    assert out.loss is not None
    assert out.aux["lora_coeffs"].shape == (2, cfg.dopa.lora_modules)


def test_lazy_cold_blocks_materialize_on_demand():
    cfg = DOPAConfig.from_yaml("configs/tiny.yaml")
    cfg.offload.lazy_cold_blocks = True
    cfg.offload.enabled = False
    model = DOPACoderN1(cfg)
    assert len(model.cold_manager.blocks) == 0
    x = torch.randint(0, cfg.model.vocab_size, (1, 16))
    _ = model(x, labels=x, force_cold=True)
    assert len(model.cold_manager._lazy_blocks) >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cold block device migration")
def test_lazy_cold_blocks_follow_cuda_model_device():
    cfg = DOPAConfig.from_yaml("configs/tiny.yaml")
    cfg.offload.lazy_cold_blocks = True
    cfg.offload.enabled = False
    cfg.offload.cold_dtype = "fp16"
    model = DOPACoderN1(cfg).to("cuda")
    x = torch.randint(0, cfg.model.vocab_size, (1, 8), device="cuda")
    out = model(x, labels=x, force_cold=True)
    assert out.logits.device.type == "cuda"
    assert any(next(block.parameters()).device.type == "cuda" for block in model.cold_manager._lazy_blocks.values())


def test_fine_grained_cold_units_forward():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    model = DOPACoderN1(cfg)
    assert model.use_fine_cold
    x = torch.randint(0, cfg.model.vocab_size, (1, 16))
    out = model(x, labels=x, force_cold=True, return_aux=True)
    assert out.logits.shape == (1, 16, cfg.model.vocab_size)
    assert int(out.aux["cold_unit_count"]) == (
        cfg.offload.cold_attention_heads_per_step + cfg.offload.cold_ffn_blocks_per_step
    )
    assert out.aux["cold_weights"].shape[-1] == int(out.aux["cold_unit_count"])
    assert torch.allclose(out.aux["cold_weights"].sum(dim=-1), torch.ones(1), atol=1e-6)
    assert out.aux["difficulty"].shape == (1, 16)


def test_fine_grained_ldp_respects_layer_budget():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.offload.cold_layer_budget_per_step = 1
    model = DOPACoderN1(cfg)
    hidden = torch.randn(1, 4, cfg.model.d_model)
    selection = model.layer_demand(hidden)
    selected_layers = {unit.layer for unit in selection.units}
    assert len(selected_layers) <= cfg.offload.cold_layer_budget_per_step
    assert model.layer_demand.visit_counts.sum() == len(selection.units)


def test_fine_grained_cold_units_load_int8(tmp_path):
    from dopa_coder_n1.model.fine_cold import ColdUnitId
    from dopa_coder_n1.model.quantization import quantize_int8_state_dict

    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.offload.cold_checkpoint_dir = str(tmp_path)
    model = DOPACoderN1(cfg)
    uid = ColdUnitId(layer=0, kind="head", index=0)
    unit = model.fine_cold_shell.store.create(uid)
    torch.save(quantize_int8_state_dict(unit.state_dict()), tmp_path / f"{uid.key}.int8.pt")
    loaded = model.fine_cold_shell.store.load(uid)
    x = torch.randn(1, 4, cfg.model.d_model)
    y = loaded(x)
    assert y.shape == x.shape


def test_fine_grained_cold_units_load_int4(tmp_path):
    from dopa_coder_n1.model.fine_cold import ColdUnitId
    from dopa_coder_n1.model.quantization import quantize_int4_state_dict

    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.offload.cold_checkpoint_dir = str(tmp_path)
    model = DOPACoderN1(cfg)
    uid = ColdUnitId(layer=0, kind="head", index=0)
    unit = model.fine_cold_shell.store.create(uid)
    torch.save(quantize_int4_state_dict(unit.state_dict()), tmp_path / f"{uid.key}.int4.pt")
    loaded = model.fine_cold_shell.store.load(uid)
    x = torch.randn(1, 4, cfg.model.d_model)
    y = loaded(x)
    assert y.shape == x.shape


def test_fine_grained_checkpoint_loads_materialized_units(tmp_path):
    from dopa_coder_n1.training.checkpoint import load_checkpoint, save_checkpoint

    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 16))
    _ = model(x, labels=x, force_cold=True)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, None, 1, cfg)
    loaded, _ = load_checkpoint(path)
    assert loaded.use_fine_cold


def test_incremental_forward_and_generate():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.hot_kv_window = 3
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 5))
    cache = None
    for pos in range(x.size(1)):
        out = model.forward_incremental(x[:, pos : pos + 1], position=pos, hot_kv_cache=cache)
        cache = out.hot_kv_cache
    assert out.logits.shape == (1, 1, cfg.model.vocab_size)
    assert all(isinstance(layer_cache, PackedLayerKV) for layer_cache in cache)
    assert all(layer_cache.shape[2] <= 3 for layer_cache in cache)
    assert any(layer_cache.has_summary for layer_cache in cache)
    y = model.generate(x, max_new_tokens=3, temperature=0.0, use_incremental=True)
    assert y.shape == (1, 8)


def test_incremental_cold_selective_kv_cache():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.hot_kv_window = 3
    cfg.dopa.cold_kv_window = 2
    cfg.dopa.cold_kv_hot_units = 2
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 4))
    hot_cache = None
    cold_cache = None
    for pos in range(x.size(1)):
        out = model.forward_incremental(
            x[:, pos : pos + 1],
            position=pos,
            hot_kv_cache=hot_cache,
            cold_kv_cache=cold_cache,
            force_cold=True,
        )
        hot_cache = out.hot_kv_cache
        cold_cache = out.cold_kv_cache
    assert cold_cache is not None
    assert 0 < len(cold_cache.cache) <= cfg.dopa.cold_kv_hot_units
    assert all(entry.shape[2] <= cfg.dopa.cold_kv_window for entry in cold_cache.cache.values())


def test_stage_freezing_supports_fine_grained_cold_shell():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    model = DOPACoderN1(cfg)
    model.freeze_base_for_stage("stage2")
    assert not any(p.requires_grad for p in model.fine_cold_shell.parameters())
    assert any(p.requires_grad for p in model.layer_demand.parameters())
    model.freeze_base_for_stage("stage1")
    assert not any(p.requires_grad for p in model.layer_demand.parameters())


def test_stage2_loss_uses_fine_grained_cold_weights():
    from dopa_coder_n1.training.stages import stage_loss

    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))
    loss, metrics = stage_loss(model, {"input_ids": x, "labels": x}, "stage2")
    assert loss.isfinite()
    assert "cold_usage" in metrics
    assert "cold_visit_mean" in metrics
    assert metrics["cold_usage"] > 0
