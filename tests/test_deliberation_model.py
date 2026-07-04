import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.training.stages import normalize_stage, stage_loss


def test_model_forward_returns_adaptive_deliberation_aux_when_enabled():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.adaptive_deliberation_enabled = True
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))

    out = model(x, labels=x, return_aux=True)

    assert out.aux["deliberation_logits"].shape == (1, 5)
    assert out.aux["deliberation_level"].shape == (1,)
    assert out.aux["task_complexity"].shape == (1,)
    assert out.aux["ambiguity_score"].shape == (1,)
    assert out.aux["process_reward"].shape == (1,)


def test_stage_deliberation_adds_scheduler_reward_and_ambiguity_losses():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.adaptive_deliberation_enabled = True
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))

    loss, metrics = stage_loss(
        model,
        {
            "input_ids": x,
            "labels": x,
            "deliberation_level_labels": torch.tensor([3]),
            "process_reward_labels": torch.tensor([0.75]),
            "ambiguity_labels": torch.tensor([0.0]),
        },
        "stage_deliberation",
    )

    assert normalize_stage("stage_deliberation") == "adaptive_deliberation"
    assert loss.isfinite()
    assert "deliberation_policy_ce" in metrics
    assert "process_reward_loss" in metrics
    assert "ambiguity_bce" in metrics
