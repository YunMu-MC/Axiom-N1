import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.model.dspark import (
    DSparkHeads,
    VerificationScheduler,
    accept_prefix_from_distributions,
)


def test_dspark_heads_emit_parallel_draft_markov_and_confidence_outputs():
    heads = DSparkHeads(d_model=16, vocab_size=23, gamma=7, markov_rank=4)
    anchor_hidden = torch.randn(2, 16)
    previous_tokens = torch.tensor([3, 5])

    out = heads(anchor_hidden, previous_tokens=previous_tokens)

    assert out.draft_logits.shape == (2, 6, 23)
    assert out.markov_logits.shape == (2, 6, 23)
    assert out.corrected_logits.shape == (2, 6, 23)
    assert out.confidence.shape == (2, 6)
    assert torch.all(out.confidence >= 0)
    assert torch.all(out.confidence <= 1)
    assert out.draft_tokens.shape == (2, 6)


def test_verification_scheduler_shortens_under_high_load_and_low_confidence():
    scheduler = VerificationScheduler(gamma=7, min_verify_tokens=1)
    confidence = torch.tensor([[0.99, 0.95, 0.80, 0.30, 0.10, 0.05]])

    low_load = scheduler(confidence, engine_load=0.10)
    high_load = scheduler(confidence, engine_load=0.95)

    assert low_load.verify_lengths.item() >= high_load.verify_lengths.item()
    assert high_load.verify_lengths.item() >= 1
    assert high_load.mask.sum().item() == high_load.verify_lengths.item()


def test_accept_prefix_stops_at_first_target_disagreement():
    draft_tokens = torch.tensor([[1, 2, 3, 4]])
    draft_log_probs = torch.log_softmax(torch.tensor([[[0.0, 8.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 8.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 8.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 8.0]]]), dim=-1)
    target_log_probs = torch.log_softmax(torch.tensor([[[0.0, 8.0, 0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 8.0, 0.0, 0.0],
                                                        [8.0, 0.0, 0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 0.0, 0.0, 8.0]]]), dim=-1)

    result = accept_prefix_from_distributions(draft_tokens, draft_log_probs, target_log_probs)

    assert result.accepted_lengths.tolist() == [2]
    assert result.accepted_tokens[0].tolist() == [1, 2]
