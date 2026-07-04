from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.data.dataset import PackedTextDataset, collate_batch
from dopa_coder_n1.data.skeleton_tokenizer import SkeletonTokenizer
from dopa_coder_n1.data.tokenizer import ByteTokenizer
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-horizon DOPA evaluation.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--config")
    parser.add_argument("--data", default=None)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompts", default=None, help="Text file with one prompt per line.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.checkpoint:
        model, raw = load_checkpoint(args.checkpoint, map_location="cpu")
        cfg = model.cfg
        checkpoint_step = int(raw.get("step", 0))
    else:
        cfg = DOPAConfig.from_yaml(args.config)
        tokenizer = ByteTokenizer()
        cfg.model.vocab_size = tokenizer.vocab_size
        model = DOPACoderN1(cfg)
        checkpoint_step = 0
    if args.data is not None:
        cfg.data.valid_path = args.data
    eval_path = cfg.data.valid_path or cfg.data.train_path
    if eval_path is None:
        raise SystemExit("Set data.valid_path/data.train_path or pass --data")
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    model.to(device)
    model.eval()
    tokenizer = ByteTokenizer()
    skeleton_tokenizer = SkeletonTokenizer(cfg.dopa.skeleton_vocab_size)
    dataset = PackedTextDataset(eval_path, tokenizer=tokenizer, seq_len=cfg.model.max_seq_len, skeleton_tokenizer=skeleton_tokenizer)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, num_workers=0, collate_fn=collate_batch)
    metrics = evaluate_loss(model, loader, device=device, max_batches=args.max_batches)
    metrics["checkpoint_step"] = checkpoint_step
    metrics["samples"] = generate_samples(
        model,
        tokenizer,
        prompts=_load_prompts(args.prompts),
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


@torch.no_grad()
def evaluate_loss(model: DOPACoderN1, loader: DataLoader, *, device: torch.device, max_batches: int) -> dict:
    losses = []
    difficulty = []
    cold_units = []
    cold_usage = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        if "skeleton" in batch:
            batch["skeleton"].token_ids = batch["skeleton"].token_ids.to(device)
        out = model(batch["input_ids"], labels=batch["labels"], skeleton=batch.get("skeleton"), return_aux=True)
        if out.loss is not None:
            losses.append(float(out.loss.detach().cpu()))
        aux = out.aux or {}
        if "difficulty" in aux:
            difficulty.append(float(aux["difficulty"].mean().detach().cpu()))
        if "cold_unit_count" in aux:
            cold_units.append(float(aux["cold_unit_count"].detach().cpu()))
        if "cold_weights" in aux:
            cold_usage.append(float(aux["cold_weights"].sum(dim=-1).mean().detach().cpu()))
    mean_loss = sum(losses) / max(1, len(losses))
    return {
        "eval_batches": len(losses),
        "loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "difficulty_mean": sum(difficulty) / max(1, len(difficulty)),
        "cold_unit_count_mean": sum(cold_units) / max(1, len(cold_units)),
        "cold_usage_mean": sum(cold_usage) / max(1, len(cold_usage)),
    }


@torch.no_grad()
def generate_samples(
    model: DOPACoderN1,
    tokenizer: ByteTokenizer,
    *,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    samples = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0, eos_id=tokenizer.eos_id, use_incremental=True)
        samples.append({"prompt": prompt, "completion": tokenizer.decode(out[0].tolist())})
    return samples


def _load_prompts(path: str | None) -> list[str]:
    if path is None:
        return ["def solve():\n    ", "class Solution:\n    def"]
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
