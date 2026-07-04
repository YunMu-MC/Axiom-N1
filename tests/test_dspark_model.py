import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.inference.dspark_decode import dspark_generate
from dopa_coder_n1.training.stages import normalize_stage, stage_loss


def test_model_forward_returns_dspark_aux_when_enabled():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.dspark_enabled = True
    cfg.dopa.dspark_gamma = 5
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))

    out = model(x, labels=x, return_aux=True)

    assert out.aux["dspark_draft_logits"].shape == (1, 4, cfg.model.vocab_size)
    assert out.aux["dspark_confidence"].shape == (1, 4)
    assert out.aux["dspark_verify_mask"].shape == (1, 4)
    assert out.aux["dspark_verify_lengths"].shape == (1,)


def test_stage_dspark_trains_draft_confidence_and_markov_heads():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.dspark_enabled = True
    cfg.dopa.dspark_gamma = 5
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))

    loss, metrics = stage_loss(
        model,
        {
            "input_ids": x,
            "labels": x,
            "dspark_accept_labels": torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        },
        "stage_dspark",
    )

    assert normalize_stage("stage_dspark") == "dspark_speculative"
    assert loss.isfinite()
    assert "dspark_draft_ce" in metrics
    assert "dspark_confidence_bce" in metrics
    assert "dspark_markov_kl" in metrics


def test_dspark_generate_uses_speculative_cycles_and_preserves_prefix():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.dspark_enabled = True
    cfg.dopa.dspark_gamma = 4
    model = DOPACoderN1(cfg)
    input_ids = torch.randint(0, cfg.model.vocab_size, (1, 4))

    out, stats = dspark_generate(model, input_ids, max_new_tokens=3, temperature=0.0)

    assert out.shape[1] >= input_ids.shape[1]
    assert torch.equal(out[:, : input_ids.shape[1]], input_ids)
    assert stats.cycles >= 1
    assert stats.proposed_tokens >= stats.accepted_tokens
