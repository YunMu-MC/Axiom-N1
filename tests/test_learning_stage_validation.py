from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.training.validation import validate_training_config


def test_validation_rejects_learning_stage_without_feature_flag():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.train.train_stage = "stage_sdit"
    errors = validate_training_config(cfg, require_data=False)
    assert any("self_driven_induction_enabled" in error for error in errors)

    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.train.train_stage = "stage5"
    errors = validate_training_config(cfg, require_data=False)
    assert any("permanent_knowledge_enabled" in error for error in errors)


def test_tiny_learning_config_accepts_learning_stages():
    cfg = DOPAConfig.from_yaml("configs/tiny_learning.yaml")

    cfg.train.train_stage = "stage_sdit"
    assert validate_training_config(cfg, require_data=False) == []

    cfg.train.train_stage = "stage5"
    assert validate_training_config(cfg, require_data=False) == []
