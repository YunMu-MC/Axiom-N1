import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.model.alignment import IntentImplementationAlignmentGate


def test_iia_gate_scores_intent_implementation_alignment():
    gate = IntentImplementationAlignmentGate(d_model=8, vocab_size=16, threshold=0.30)
    intent = torch.randn(2, 5, 8)
    implementation = torch.randn(2, 7, 8)

    result = gate(intent, implementation)

    assert result.score.shape == (2,)
    assert result.triggered.shape == (2,)
    assert torch.all(result.score >= 0)
    assert torch.all(result.score <= 1)


def test_iia_gate_soft_adjusts_logits_only_below_threshold():
    gate = IntentImplementationAlignmentGate(d_model=4, vocab_size=6, threshold=0.30, soft_strength=2.0)
    gate.set_token_drift_bias({3: 1.5, 4: 0.5})
    logits = torch.zeros(2, 6)

    adjusted = gate.apply_logit_bias(logits, torch.tensor([0.10, 0.90]))

    assert adjusted[0, 3] < adjusted[0, 4] < adjusted[0, 0]
    assert torch.allclose(adjusted[1], logits[1])
