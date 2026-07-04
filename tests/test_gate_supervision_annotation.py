import json

import pytest

torch = pytest.importorskip("torch")

from scripts.annotate_gate_supervision import (
    annotate_record_from_losses,
    labels_from_hot_losses,
    iter_jsonl_paths,
)


def test_labels_from_hot_losses_selects_hot_hard_tokens_without_teacher_losses():
    losses = torch.tensor([0.2, 3.1, 7.0, float("nan"), 2.9])

    labels, mask = labels_from_hot_losses(losses, hot_threshold=3.0, max_positive_rate=1.0)

    assert labels.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert mask.tolist() == [True, True, True, False, True]


def test_labels_from_hot_losses_caps_positive_rate_by_hardest_tokens():
    losses = torch.tensor([5.0, 4.0, 3.5, 3.2])

    labels, mask = labels_from_hot_losses(losses, hot_threshold=3.0, max_positive_rate=0.5)

    assert mask.tolist() == [True, True, True, True]
    assert labels.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_annotate_record_from_losses_adds_gate_fields_and_metadata():
    record = {"text": "User: hello\n\nAssistant: hi", "metadata": {"source": "unit"}}
    losses = torch.tensor([0.1, 4.2, 0.3, 6.0])

    annotated = annotate_record_from_losses(
        record,
        token_losses=losses,
        encoded_len=4,
        hot_threshold=3.0,
        max_positive_rate=1.0,
        config_name="configs/tiny_unit.yaml",
        checkpoint="",
    )

    assert annotated["external_knowledge_labels"] == [0.0, 1.0, 0.0, 1.0]
    assert annotated["hot_token_losses"] == [0.1, 4.2, 0.3, 6.0]
    assert annotated["metadata"]["gate_supervision"]["source"] == "hot_ce"


def test_iter_jsonl_paths_keeps_single_file_and_directory_order(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "nested" / "b.jsonl"
    second.parent.mkdir()
    first.write_text(json.dumps({"text": "a"}) + "\n", encoding="utf-8")
    second.write_text(json.dumps({"text": "b"}) + "\n", encoding="utf-8")

    assert list(iter_jsonl_paths([first])) == [first]
    assert list(iter_jsonl_paths([tmp_path])) == [first, second]
