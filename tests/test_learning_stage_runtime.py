import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.training.stages import stage_loss
from dopa_coder_n1.training.validation import validate_training_config


def test_validation_accepts_learning_stages_when_enabled():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.self_driven_induction_enabled = True
    cfg.train.train_stage = "stage_sdit"
    assert not validate_training_config(cfg, require_data=False)

    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.permanent_knowledge_enabled = True
    cfg.train.train_stage = "stage5"
    assert not validate_training_config(cfg, require_data=False)


def test_stage_sdit_can_synthesize_variants_from_plain_lm_batch():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.self_driven_induction_enabled = True
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))

    loss, metrics = stage_loss(model, {"input_ids": x, "labels": x}, "stage_sdit")

    assert loss.isfinite()
    assert metrics["sdit_consistency"].isfinite()
    assert metrics["sdit_transfer"].isfinite()
    assert metrics["sdit_reverse"].isfinite()


def test_stage5_defaults_to_query_action_for_plain_lm_batch():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.permanent_knowledge_enabled = True
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))

    loss, metrics = stage_loss(model, {"input_ids": x, "labels": x}, "stage5")

    assert loss.isfinite()
    assert "knowledge_policy_ce" in metrics
