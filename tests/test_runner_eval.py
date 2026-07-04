import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.data.dataset import PackedTextDataset, collate_batch
from dopa_coder_n1.data.skeleton_tokenizer import SkeletonTokenizer
from dopa_coder_n1.data.tokenizer import ByteTokenizer
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.training.cold_units import writeback_materialized_cold_units
from dopa_coder_n1.training.runner import train_one_stage


def test_train_runner_writes_metrics_and_checkpoint(tmp_path):
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.train.max_steps = 1
    cfg.train.checkpoint_every = 1
    cfg.train.log_every = 1
    cfg.data.train_path = "data/toy.jsonl"
    metrics = train_one_stage(cfg, out_dir=tmp_path, device=torch.device("cpu"))
    assert metrics["step"] == 1
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "metrics.jsonl").exists()


def test_writeback_materialized_cold_units(tmp_path):
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    model = DOPACoderN1(cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))
    _ = model(x, labels=x, force_cold=True)
    count = writeback_materialized_cold_units(model, tmp_path, fmt="int4")
    assert count > 0
    assert list(tmp_path.glob("*.int4.pt"))


def test_eval_loader_path_smoke():
    from scripts.evaluate import evaluate_loss

    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    tokenizer = ByteTokenizer()
    cfg.model.vocab_size = tokenizer.vocab_size
    model = DOPACoderN1(cfg)
    dataset = PackedTextDataset(
        "data/toy.jsonl",
        tokenizer=tokenizer,
        seq_len=cfg.model.max_seq_len,
        skeleton_tokenizer=SkeletonTokenizer(cfg.dopa.skeleton_vocab_size),
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_batch)
    metrics = evaluate_loss(model, loader, device=torch.device("cpu"), max_batches=1)
    assert metrics["eval_batches"] == 1
    assert metrics["loss"] > 0
