import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.metacognition import FailurePredictionGate, should_trigger_posthoc_learning


def test_failure_prediction_gate_outputs_probability_per_sample():
    gate = FailurePredictionGate(d_model=8)
    hidden = torch.randn(3, 5, 8)

    probability = gate(hidden)

    assert probability.shape == (3,)
    assert torch.all(probability >= 0)
    assert torch.all(probability <= 1)


def test_model_forward_returns_failure_probability_when_enabled():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.metacognition_enabled = True
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))

    out = model(x, labels=x, return_aux=True)

    assert out.aux["failure_probability"].shape == (1,)
    assert out.aux["failure_overconfidence"].shape == (1,)


def test_posthoc_learning_triggers_when_execution_failed_but_model_was_confident():
    assert should_trigger_posthoc_learning(
        predicted_failure_probability=torch.tensor([0.05]),
        execution_failed=True,
        overconfidence_threshold=0.20,
    )
    assert not should_trigger_posthoc_learning(
        predicted_failure_probability=torch.tensor([0.80]),
        execution_failed=True,
        overconfidence_threshold=0.20,
    )
