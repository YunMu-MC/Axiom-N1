import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.training.stages import normalize_stage, stage_loss


def test_stage5_trains_knowledge_operation_policy_head():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.permanent_knowledge_enabled = True
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))

    out = model(x, labels=x, return_aux=True)

    assert out.aux["knowledge_policy_logits"].shape == (1, 5)

    loss, metrics = stage_loss(
        model,
        {
            "input_ids": x,
            "labels": x,
            "knowledge_action_labels": torch.tensor([0]),
        },
        "stage5",
    )

    assert normalize_stage("stage5") == "knowledge_management"
    assert loss.isfinite()
    assert "knowledge_policy_ce" in metrics
