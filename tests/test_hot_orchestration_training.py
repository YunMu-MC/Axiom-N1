import json

import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.utils.data import DataLoader

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.data.dataset import PackedTextDataset, collate_batch
from dopa_coder_n1.data.skeleton_tokenizer import SkeletonTokenizer
from dopa_coder_n1.data.tokenizer import ByteTokenizer
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.fine_cold import FineGrainedLayerDemandPredictor
from dopa_coder_n1.training.hot_orchestration import derive_external_knowledge_labels
from dopa_coder_n1.training.stages import stage_loss


def test_external_knowledge_labels_use_hot_hard_teacher_easy_margin():
    hot = torch.tensor([[0.4, 3.2, 4.5, 2.8, 6.0]])
    teacher = torch.tensor([[0.3, 0.4, 3.0, 0.2, float("nan")]])
    labels, mask = derive_external_knowledge_labels(
        hot,
        teacher,
        hot_threshold=3.0,
        teacher_easy_threshold=1.0,
        margin=1.0,
    )

    assert labels.tolist() == [[0.0, 1.0, 0.0, 0.0, 0.0]]
    assert mask.tolist() == [[True, True, True, True, False]]


def test_stage_loss_jointly_trains_difficulty_gate_and_ldp_from_teacher_signal():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.dopa.external_knowledge_gate_loss_weight = 0.25
    cfg.dopa.external_knowledge_ldp_loss_weight = 0.15
    model = DOPACoderN1(cfg)
    input_ids = torch.randint(4, cfg.model.vocab_size, (2, 10))
    teacher = torch.full((2, 10), 0.2)
    hot = torch.full((2, 10), 0.4)
    hot[:, 3:6] = 4.0
    batch = {
        "input_ids": input_ids,
        "labels": input_ids,
        "teacher_token_losses": teacher,
        "hot_token_losses": hot,
    }

    loss, metrics = stage_loss(model, batch, "stage1")

    assert loss.requires_grad
    assert "external_knowledge_bce" in metrics
    assert "ldp_need_bce" in metrics
    assert metrics["external_knowledge_positive_rate"] > 0


def test_packed_dataset_collates_external_knowledge_supervision(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "abcdefghi",
                "skeleton": {"name": "toy", "params": [], "steps": [{"op": "return"}]},
                "external_knowledge_labels": [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                "teacher_token_losses": [0.1] * 12,
                "hot_token_losses": [4.0] * 12,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = PackedTextDataset(
        path,
        tokenizer=ByteTokenizer(rust_backend=None),
        seq_len=8,
        skeleton_tokenizer=SkeletonTokenizer(128, rust_backend=None),
        skeleton_len=16,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate_batch)))

    assert batch["external_knowledge_labels"].shape == (1, 8)
    assert batch["teacher_token_losses"].shape == (1, 8)
    assert batch["hot_token_losses"].shape == (1, 8)
    assert batch["skeleton"].token_ids.shape == (1, 16)



def test_ldp_routes_with_active_mask_but_returns_raw_logits_for_supervision():
    predictor = FineGrainedLayerDemandPredictor(
        d_model=4,
        cold_layers=2,
        n_heads=1,
        ffn_subblocks=1,
        layer_budget=1,
        head_budget=1,
        ffn_budget=1,
        coverage_penalty=0.5,
    )
    predictor.net = nn.Identity()
    hidden = torch.tensor([[[3.0, 3.0, 0.0, 0.0], [0.0, 0.0, 100.0, 100.0]]])
    active = torch.tensor([[True, False]])

    selection = predictor(hidden, active=active)

    assert {unit.layer for unit in selection.units} == {0}
    assert selection.logits[0, 1, 2].item() == 100.0



def test_collate_keeps_partial_external_knowledge_supervision(tmp_path):
    path = tmp_path / "train.jsonl"
    long_text = "x" * 140
    path.write_text(
        json.dumps(
            {
                "text": long_text,
                "external_knowledge_labels": [1.0] * 16,
                "hot_token_losses": [4.0] * 16,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = PackedTextDataset(
        path,
        tokenizer=ByteTokenizer(rust_backend=None),
        seq_len=32,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_batch)))

    assert batch["external_knowledge_labels"].shape == (2, 32)
    assert batch["external_knowledge_labels"][0].max() == 1.0
    assert torch.all(batch["external_knowledge_labels"][1] == -1.0)
    assert batch["hot_token_losses"].shape == (2, 32)
    assert torch.isnan(batch["hot_token_losses"][1]).all()



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression requires CUDA")
def test_stage_loss_external_knowledge_supervision_is_cuda_autocast_safe():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    model = DOPACoderN1(cfg).cuda()
    input_ids = torch.randint(4, cfg.model.vocab_size, (1, 16), device="cuda")
    batch = {
        "input_ids": input_ids,
        "labels": input_ids,
        "external_knowledge_labels": torch.ones((1, 16), device="cuda"),
    }

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, metrics = stage_loss(model, batch, "stage1")

    assert loss.requires_grad
    assert "external_knowledge_bce" in metrics
