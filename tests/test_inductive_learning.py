import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.inductive_learning import SelfDrivenInductiveLoss
from dopa_coder_n1.training.stages import normalize_stage, stage_loss


def test_sdit_loss_reports_consistency_transfer_and_reverse_prediction():
    loss_fn = SelfDrivenInductiveLoss(consistency_weight=0.2, transfer_weight=0.3, reverse_weight=0.4)
    orig = torch.randn(2, 8)
    var_a = orig + 0.01
    var_b = torch.randn(2, 8)
    reverse_logits = torch.randn(2, 5, 11)
    rule_labels = torch.randint(0, 11, (2, 5))

    total, metrics = loss_fn(
        orig,
        var_a,
        var_b,
        reverse_logits=reverse_logits,
        rule_labels=rule_labels,
    )

    assert total.isfinite()
    assert metrics["sdit_consistency"] < metrics["sdit_transfer"]
    assert metrics["sdit_reverse"] > 0


def test_stage_sdit_uses_variants_and_rule_labels():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.self_driven_induction_enabled = True
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))
    variant_a = x.clone()
    variant_a[:, -1] = (variant_a[:, -1] + 1) % cfg.model.vocab_size
    variant_b = torch.flip(x, dims=[1])
    rule_labels = torch.randint(0, cfg.model.vocab_size, (1, 8))

    loss, metrics = stage_loss(
        model,
        {
            "input_ids": x,
            "labels": x,
            "variant_a_input_ids": variant_a,
            "variant_b_input_ids": variant_b,
            "rule_labels": rule_labels,
        },
        "stage_sdit",
    )

    assert normalize_stage("stage_sdit") == "self_driven_induction"
    assert loss.isfinite()
    assert "sdit_consistency" in metrics
    assert "sdit_transfer" in metrics
    assert "sdit_reverse" in metrics
