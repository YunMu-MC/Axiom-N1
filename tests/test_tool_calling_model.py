import json

import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.data.dataset import PackedTextDataset, collate_batch
from dopa_coder_n1.data.tokenizer import ByteTokenizer
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.training.stages import stage_loss
from dopa_coder_n1.training.validation import validate_training_config


def _tool_cfg() -> DOPAConfig:
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.tool_calling_enabled = True
    cfg.dopa.tool_action_count = 8
    cfg.dopa.tool_query_dim = 16
    cfg.model.vocab_size = 128
    return cfg


def test_model_forward_returns_tool_calling_aux_when_enabled():
    cfg = _tool_cfg()
    model = DOPACoderN1(cfg)
    input_ids = torch.randint(0, cfg.model.vocab_size, (2, 6))

    out = model(input_ids, labels=input_ids, return_aux=True)

    assert out.aux["tool_need_logits"].shape == (2, cfg.dopa.tool_action_count)
    assert out.aux["tool_argument_validity"].shape == (2,)
    assert out.aux["tool_query_embedding"].shape == (2, cfg.dopa.tool_query_dim)


def test_stage_tool_calling_trains_need_argument_and_query_heads():
    cfg = _tool_cfg()
    model = DOPACoderN1(cfg)
    input_ids = torch.randint(0, cfg.model.vocab_size, (2, 7))
    batch = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "tool_need_labels": torch.tensor([1, 3]),
        "tool_argument_valid_labels": torch.tensor([1.0, 0.0]),
        "tool_query_targets": torch.randn(2, cfg.dopa.tool_query_dim),
    }

    loss, metrics = stage_loss(model, batch, "stage_tool_calling")

    assert loss.requires_grad
    assert "tool_need_ce" in metrics
    assert "tool_argument_bce" in metrics
    assert "tool_query_cosine" in metrics


def test_jsonl_dataset_carries_tool_calling_labels(tmp_path):
    row = {
        "text": "Use fs.read_text to inspect a file.",
        "tool_need_label": 2,
        "tool_argument_valid_label": 1.0,
        "tool_query_target": [0.25, -0.25, 0.5, -0.5],
    }
    path = tmp_path / "tool.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = PackedTextDataset(path, ByteTokenizer(rust_backend=None), seq_len=8)
    item = next(iter(dataset))

    assert item["tool_need_labels"].item() == 2
    assert item["tool_argument_valid_labels"].item() == 1.0
    assert item["tool_query_targets"].shape == (4,)

    batch = collate_batch([item, item])
    assert batch["tool_need_labels"].shape == (2,)
    assert batch["tool_query_targets"].shape == (2, 4)


def test_tool_calling_stage_requires_enabled_config():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.train.train_stage = "stage_tool_calling"
    cfg.dopa.tool_calling_enabled = False

    errors = validate_training_config(cfg, require_data=False)

    assert any("tool_calling_enabled" in item for item in errors)
