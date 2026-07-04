from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.data.dataset import PackedTextDataset, collate_batch
from dopa_coder_n1.data.skeleton_tokenizer import SkeletonTokenizer
from dopa_coder_n1.data.tokenizer import ByteTokenizer
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.shadow import rotate_shadow_masks
from dopa_coder_n1.training.checkpoint import load_checkpoint, save_checkpoint
from dopa_coder_n1.training.optim import CosineWithWarmup, build_optimizer
from dopa_coder_n1.training.stages import stage_loss


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_training_objects(
    cfg: DOPAConfig,
    *,
    device: torch.device,
    resume: str | Path | None = None,
) -> tuple[DOPACoderN1, torch.optim.Optimizer, CosineWithWarmup, int]:
    if resume is not None:
        model, raw = load_checkpoint(resume, map_location="cpu")
        cfg.model.vocab_size = model.cfg.model.vocab_size
        same_stage = raw.get("config", {}).get("train", {}).get("train_stage") == cfg.train.train_stage
        start_step = int(raw.get("step", 0)) if same_stage else 0
    else:
        model = DOPACoderN1(cfg)
        raw = {}
        start_step = 0
    model.freeze_base_for_stage(cfg.train.train_stage)
    model.to(device)
    if cfg.offload.enabled and model.cold_manager is not None:
        model.cold_manager.offload_all()
    optimizer = build_optimizer(
        model,
        cfg.train.learning_rate,
        cfg.train.weight_decay,
        state_device=cfg.train.optimizer_state_device,
    )
    can_load_optimizer = resume is not None and raw.get("optimizer") is not None and start_step > 0
    if can_load_optimizer:
        optimizer.load_state_dict(raw["optimizer"])
    scheduler = CosineWithWarmup(optimizer, cfg.train.warmup_steps, cfg.train.max_steps)
    scheduler.step_num = start_step
    return model, optimizer, scheduler, start_step


def build_loader(cfg: DOPAConfig, tokenizer: ByteTokenizer, skeleton_tokenizer: SkeletonTokenizer) -> DataLoader:
    if cfg.data.train_path is None:
        raise ValueError("Set data.train_path in config or pass --data")
    dataset = PackedTextDataset(
        cfg.data.train_path,
        tokenizer=tokenizer,
        seq_len=cfg.model.max_seq_len,
        skeleton_tokenizer=skeleton_tokenizer,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_batch,
    )


def train_one_stage(
    cfg: DOPAConfig,
    *,
    out_dir: str | Path,
    device: torch.device,
    resume: str | Path | None = None,
) -> dict[str, float | int]:
    torch.manual_seed(cfg.train.seed)
    cfg.offload.device = str(device)
    tokenizer = ByteTokenizer()
    cfg.model.vocab_size = tokenizer.vocab_size
    skeleton_tokenizer = SkeletonTokenizer(cfg.dopa.skeleton_vocab_size)
    model, optimizer, scheduler, step = build_training_objects(cfg, device=device, resume=resume)
    loader = build_loader(cfg, tokenizer, skeleton_tokenizer)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(out / "config.yaml")
    tokenizer.save(out / "byte_tokenizer.json")
    metrics_path = out / "metrics.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_metrics: dict[str, float | int] = {"step": step}
    progress = tqdm(total=cfg.train.max_steps, initial=step, desc=f"train:{cfg.train.train_stage}")
    while step < cfg.train.max_steps:
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if "skeleton" in batch:
                batch["skeleton"].token_ids = batch["skeleton"].token_ids.to(device)
            with autocast_context(device, cfg.train.precision):
                loss, metrics = stage_loss(model, batch, cfg.train.train_stage)
                scaled_loss = loss / cfg.train.grad_accum_steps
            scaled_loss.backward()
            if (step + 1) % cfg.train.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            last_metrics = {"step": step, "stage": cfg.train.train_stage}
            last_metrics.update({k: float(v.detach().cpu()) for k, v in metrics.items()})
            if step % cfg.train.log_every == 0:
                progress.set_postfix_str(" ".join(f"{k}={v:.4f}" for k, v in last_metrics.items() if isinstance(v, float)))
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(last_metrics, ensure_ascii=False) + "\n")
            if step % cfg.train.checkpoint_every == 0:
                save_checkpoint(out / f"step_{step}.pt", model, optimizer, step, cfg)
            if cfg.train.shadow_rotate_every and step % cfg.train.shadow_rotate_every == 0:
                rotate_shadow_masks(model, preserve_delta=True)
            progress.update(1)
            if step >= cfg.train.max_steps:
                break
    save_checkpoint(out / "last.pt", model, optimizer, step, cfg)
    progress.close()
    return last_metrics
