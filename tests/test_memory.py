import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.quantization import (
    dequantize_int4_state_dict,
    dequantize_int8_state_dict,
    quantize_int4_state_dict,
    quantize_int8_state_dict,
)
from dopa_coder_n1.utils.memory import estimate_memory


def test_local_budget_fits_8gb_16gb():
    cfg = DOPAConfig.from_yaml("configs/local_8gb_16gb.yaml")
    report = estimate_memory(cfg)
    assert report.fits_vram
    assert report.fits_ram


def test_64b_config_uses_lazy_cold_blocks():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    assert cfg.offload.lazy_cold_blocks
    assert cfg.offload.quantized_cold
    report = estimate_memory(cfg)
    assert report.cpu_ram_gb <= cfg.offload.ram_budget_gb


def test_int8_quantization_roundtrip_shape():
    state = {"w": torch.randn(8, 4), "n": torch.tensor([1, 2, 3])}
    packed = quantize_int8_state_dict(state)
    restored = dequantize_int8_state_dict(packed)
    assert restored["w"].shape == state["w"].shape
    assert restored["n"].equal(state["n"])


def test_int4_quantization_roundtrip_shape():
    state = {"w": torch.randn(8, 4), "n": torch.tensor([1, 2, 3])}
    packed = quantize_int4_state_dict(state)
    restored = dequantize_int4_state_dict(packed)
    assert restored["w"].shape == state["w"].shape
    assert restored["n"].equal(state["n"])



def test_memory_estimate_accounts_for_hra_and_persistent_memory_budget():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    report = estimate_memory(cfg)
    assert report.hra_transient_kv_gb > 0.0
    assert report.hra_level2_gpu_gb < 0.01
    assert report.persistent_memory_disk_gb == 10.0
    assert report.fits_vram
    assert report.fits_ram



def test_memory_estimate_accounts_for_iia_gate_overhead():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    report = estimate_memory(cfg)
    assert report.alignment_overhead_gb > 0
    assert report.alignment_overhead_gb < 0.001
    assert report.fits_vram



def test_memory_estimate_accounts_for_learning_modules():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    report = estimate_memory(cfg)
    assert report.metacognition_overhead_gb > 0
    assert report.pkb_cpu_ram_gb < 0.01
    assert report.pkb_disk_gb < 0.02
    assert report.fits_vram



def test_memory_estimate_accounts_for_adaptive_deliberation():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    report = estimate_memory(cfg)
    assert 0 < report.deliberation_overhead_gb < 0.004
    assert 0 < report.deliberation_cpu_ram_gb <= 0.10
    assert report.fits_vram
    assert report.fits_ram



def test_memory_estimate_accounts_for_dspark_heads_under_40mb():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    report = estimate_memory(cfg)
    assert 0 < report.dspark_overhead_gb < 0.04
    assert report.fits_vram
