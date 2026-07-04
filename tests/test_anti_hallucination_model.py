import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.alignment import IntentImplementationAlignmentGate
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.training.stages import normalize_stage, stage_loss


def test_model_forward_returns_alignment_aux_when_intent_is_provided():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.anti_hallucination_enabled = True
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))
    intent = torch.randint(0, cfg.model.vocab_size, (1, 4))

    out = model(x, labels=x, intent_ids=intent, return_aux=True)

    assert isinstance(model.intent_alignment_gate, IntentImplementationAlignmentGate)
    assert out.aux["alignment_score"].shape == (1,)
    assert out.aux["alignment_triggered"].shape == (1,)


def test_model_alignment_soft_bias_can_change_logits_for_drift_tokens():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.anti_hallucination_enabled = True
    cfg.dopa.alignment_threshold = 1.0
    cfg.dopa.alignment_soft_strength = 4.0
    model = DOPACoderN1(cfg)
    model.intent_alignment_gate.set_token_drift_bias({5: 2.0})
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))
    intent = torch.randint(0, cfg.model.vocab_size, (1, 4))

    aligned = model(x, labels=x, intent_ids=intent, return_aux=True)
    model.cfg.dopa.anti_hallucination_enabled = False
    raw = model(x, labels=x, intent_ids=intent, return_aux=True)

    assert torch.all(aligned.logits[..., 5] <= raw.logits[..., 5])


def test_stage2_5_adds_alignment_bce_loss():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.anti_hallucination_enabled = True
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))
    intent = torch.randint(0, cfg.model.vocab_size, (1, 4))

    loss, metrics = stage_loss(
        model,
        {
            "input_ids": x,
            "labels": x,
            "intent_ids": intent,
            "alignment_labels": torch.ones(1),
        },
        "stage2_5",
    )

    assert normalize_stage("stage2_5") == "intent_alignment"
    assert loss.isfinite()
    assert "alignment_bce" in metrics


def test_stage2_5_freezes_base_and_keeps_iia_gate_trainable():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.anti_hallucination_enabled = True
    model = DOPACoderN1(cfg)

    model.freeze_base_for_stage("stage2_5")

    assert not any(p.requires_grad for p in model.hot_layers.parameters())
    assert any(p.requires_grad for p in model.intent_alignment_gate.parameters())
