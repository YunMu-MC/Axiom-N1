from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Iterator

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.data.tokenizer import ByteTokenizer
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.training.hot_orchestration import shifted_token_losses


def labels_from_hot_losses(
    losses: torch.Tensor,
    *,
    hot_threshold: float,
    max_positive_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = losses.float()
    valid = torch.isfinite(values)
    labels = (valid & (values >= hot_threshold)).to(torch.float32)
    if 0.0 < max_positive_rate < 1.0 and int(labels.sum().item()) > 0:
        valid_values = values[valid]
        if valid_values.numel() > 0:
            keep = max(1, int(math.floor(float(valid_values.numel()) * max_positive_rate)))
            keep = min(keep, valid_values.numel())
            threshold = torch.topk(valid_values, k=keep).values.min()
            labels = (valid & (values >= threshold) & (values >= hot_threshold)).to(torch.float32)
    return labels, valid


def annotate_record_from_losses(
    record: dict,
    *,
    token_losses: torch.Tensor,
    encoded_len: int,
    hot_threshold: float,
    max_positive_rate: float,
    config_name: str,
    checkpoint: str,
) -> dict:
    labels, valid = labels_from_hot_losses(
        token_losses,
        hot_threshold=hot_threshold,
        max_positive_rate=max_positive_rate,
    )
    labels = _align_1d(labels, encoded_len, fill_value=0.0)
    losses = _align_1d(token_losses.float(), encoded_len, fill_value=float("nan"))
    out = dict(record)
    out["external_knowledge_labels"] = [float(x) for x in labels.tolist()]
    out["hot_token_losses"] = [_json_float(x) for x in losses.tolist()]
    metadata = dict(out.get("metadata") or {})
    metadata["gate_supervision"] = {
        "source": "hot_ce",
        "config": config_name,
        "checkpoint": checkpoint,
        "hot_threshold": hot_threshold,
        "max_positive_rate": max_positive_rate,
        "valid_tokens": int(valid.sum().item()),
        "positive_tokens": int(labels.sum().item()),
    }
    out["metadata"] = metadata
    return out


def iter_jsonl_paths(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_file() and path.suffix == ".jsonl":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.jsonl"))


def compute_hot_token_losses(
    model: DOPACoderN1,
    tokenizer: ByteTokenizer,
    text: str,
    *,
    device: torch.device,
    max_tokens: int,
) -> tuple[torch.Tensor, int]:
    ids = tokenizer.encode(text + "\n", add_bos=True)
    if len(ids) < 2:
        return torch.empty(0), len(ids)
    if max_tokens > 0:
        ids = ids[:max_tokens]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids, return_aux=True, return_hot_logits=True)
        losses = shifted_token_losses(output.aux["hot_logits"], input_ids).squeeze(0).detach().cpu()
    losses = torch.cat([losses, torch.full((1,), float("nan"))], dim=0)
    return losses, len(ids)


def annotate_files(
    *,
    inputs: list[Path],
    out_dir: Path,
    config_path: Path,
    checkpoint_path: Path | None,
    limit: int,
    hot_threshold: float,
    max_positive_rate: float,
    max_tokens: int,
    device: torch.device,
) -> dict:
    cfg = DOPAConfig.from_yaml(config_path)
    cfg.offload.device = str(device)
    model = DOPACoderN1(cfg).to(device)
    model.eval()
    checkpoint_label = ""
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
        checkpoint_label = str(checkpoint_path)
    tokenizer = ByteTokenizer()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "records": 0,
        "positive_tokens": 0,
        "valid_tokens": 0,
        "out_dir": str(out_dir),
        "config": str(config_path),
        "checkpoint": checkpoint_label,
    }
    for source_path in iter_jsonl_paths(inputs):
        out_path = out_dir / source_path.name
        if out_path.exists():
            stem = source_path.stem
            suffix = source_path.suffix
            out_path = out_dir / f"{stem}-{_short_parent(source_path)}{suffix}"
        with source_path.open("r", encoding="utf-8", errors="ignore") as src, out_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as dst:
            for line in src:
                if limit > 0 and summary["records"] >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(record.get("text") or record.get("solution") or "").strip()
                if not text:
                    continue
                losses, encoded_len = compute_hot_token_losses(
                    model,
                    tokenizer,
                    text,
                    device=device,
                    max_tokens=max_tokens,
                )
                annotated = annotate_record_from_losses(
                    record,
                    token_losses=losses,
                    encoded_len=encoded_len,
                    hot_threshold=hot_threshold,
                    max_positive_rate=max_positive_rate,
                    config_name=str(config_path),
                    checkpoint=checkpoint_label,
                )
                meta = annotated["metadata"]["gate_supervision"]
                summary["records"] += 1
                summary["positive_tokens"] += int(meta["positive_tokens"])
                summary["valid_tokens"] += int(meta["valid_tokens"])
                dst.write(json.dumps(annotated, ensure_ascii=False, sort_keys=True) + "\n")
        if limit > 0 and summary["records"] >= limit:
            break
    (out_dir / "gate_annotation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate JSONL records with hot-core external-knowledge gate labels.")
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "gate_supervision")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "local_8gb_16gb.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hot-threshold", type=float, default=3.0)
    parser.add_argument("--max-positive-rate", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not (0.0 < args.max_positive_rate <= 1.0):
        raise RuntimeError("--max-positive-rate must be in (0, 1].")
    device = _resolve_device(args.device)
    summary = annotate_files(
        inputs=args.input,
        out_dir=args.out_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        limit=args.limit,
        hot_threshold=args.hot_threshold,
        max_positive_rate=args.max_positive_rate,
        max_tokens=args.max_tokens,
        device=device,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _align_1d(values: torch.Tensor, length: int, *, fill_value: float) -> torch.Tensor:
    values = values.flatten().float()
    if values.numel() == length:
        return values
    if values.numel() > length:
        return values[:length]
    pad = torch.full((length - values.numel(),), fill_value, dtype=torch.float32)
    return torch.cat([values, pad], dim=0)


def _json_float(value: float) -> float | None:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return round(float(value), 6)


def _short_parent(path: Path) -> str:
    parent = path.parent.name or "root"
    return "".join(ch if ch.isalnum() else "_" for ch in parent)[-24:]


if __name__ == "__main__":
    raise SystemExit(main())
