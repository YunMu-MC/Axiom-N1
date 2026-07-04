import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.fine_cold import ColdUnitId
from dopa_coder_n1.model.shadow import ShadowLinear, rotate_shadow_masks


def test_shadow_mask_rotation_changes_mask():
    base = torch.nn.Linear(8, 4, bias=False)
    shadow = ShadowLinear(base, density=0.5)
    before = shadow.mask.clone()
    count = rotate_shadow_masks(shadow, preserve_delta=True)
    assert count == 1
    assert not torch.equal(before, shadow.mask)
    assert shadow.shadow_int8.dtype == torch.int8


def test_hot_shadow_can_store_base_as_int4():
    base = torch.nn.Linear(8, 4, bias=False)
    shadow = ShadowLinear(base, density=0.5, base_quantization="int4")
    assert not hasattr(shadow, "weight")
    assert shadow.packed_weight.dtype == torch.uint8
    x = torch.randn(2, 8)
    y = shadow(x)
    assert y.shape == (2, 4)


def test_dopa_auto_injects_hot_and_lazy_cold_shadow_linears():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    model = DOPACoderN1(cfg)
    assert model.shadow_linears > 0
    hot_shadows = [module for module in model.hot_layers.modules() if isinstance(module, ShadowLinear)]
    assert hot_shadows
    assert any(module.base_quantization == "int4" for module in hot_shadows)
    unit = model.fine_cold_shell.store.load(ColdUnitId(layer=0, kind="head", index=0))
    assert any(isinstance(module, ShadowLinear) for module in unit.modules())
