import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.model.deliberation import (
    AdaptiveDeliberationScheduler,
    AmbiguityDetector,
    ThoughtLandmarkStore,
    parse_requested_deliberation_level,
    should_escalate_level,
)


def test_scheduler_uses_confidence_complexity_and_user_override():
    scheduler = AdaptiveDeliberationScheduler(d_model=8, hidden_dim=4)
    hidden = torch.randn(2, 5, 8)

    out = scheduler(
        hidden,
        curiosity_confidence=torch.tensor([0.95, 0.05]),
        failure_probability=torch.tensor([0.02, 0.90]),
        task_complexity=torch.tensor([0.10, 1.00]),
    )

    assert out.logits.shape == (2, 5)
    assert out.complexity_score.shape == (2,)
    assert out.selected_level[0] <= 1
    assert out.selected_level[1] >= 3

    override = scheduler(
        hidden,
        curiosity_confidence=torch.ones(2),
        failure_probability=torch.zeros(2),
        task_complexity=torch.zeros(2),
        requested_level="ultra",
    )
    assert override.selected_level.tolist() == [3, 3]


def test_deliberation_level_parser_and_dynamic_escalation():
    assert parse_requested_deliberation_level("please use x-open thinking") == 4
    assert parse_requested_deliberation_level("deep think carefully") == 2
    assert parse_requested_deliberation_level("fast answer") == 0
    assert parse_requested_deliberation_level("no explicit hint") is None
    assert should_escalate_level(1, verification_failed=True) == 2
    assert should_escalate_level(4, new_information=True) == 4


def test_ambiguity_detector_scores_flat_token_distribution_higher():
    detector = AmbiguityDetector(d_model=8)
    hidden = torch.zeros(2, 4, 8)
    flat_logits = torch.zeros(2, 4, 16)
    peaked_logits = flat_logits.clone()
    peaked_logits[:, :, 0] = 10.0

    flat = detector(hidden, token_logits=flat_logits)
    peaked = detector(hidden, token_logits=peaked_logits)

    assert flat.shape == (2,)
    assert torch.all(flat > peaked)
    assert detector.should_ask_clarification(flat, threshold=0.50).all()


def test_thought_landmark_store_limits_landmarks_and_keeps_tool_refs_on_cpu():
    store = ThoughtLandmarkStore(max_landmarks=3, max_landmark_tokens=5)
    ref = store.add_tool_result("long compiler log", "line\n" * 200)
    for idx in range(5):
        store.add(
            text=f"candidate plan {idx} keeps only a compact landmark",
            importance=float(idx),
            source="tree_search",
            detail_ref=ref if idx == 4 else None,
        )

    landmarks = store.list_landmarks()
    assert len(landmarks) == 3
    assert landmarks[-1].detail_ref == ref
    assert store.read_tool_result(ref).startswith("line")
    assert all(len(point.text.split()) <= 5 for point in landmarks)
