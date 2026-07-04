from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.training.cold_units import writeback_materialized_cold_units
from dopa_coder_n1.training.checkpoint import load_checkpoint
from dopa_coder_n1.training.runner import train_one_stage
from dopa_coder_n1.training.validation import assert_valid_training_config

DEFAULT_STAGES = ["stage1", "stage2", "stage3", "stage3_5"]


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end final DOPA training pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--raw-data", default=None, help="Optional raw data file/dir to split into train/valid.")
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    parser.add_argument("--steps", default=None, help="Comma-separated max steps per stage.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--writeback-format", choices=["int4", "int8", "fp32", "none"], default="int4")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = DOPAConfig.from_yaml(args.config)
    if args.raw_data is not None:
        prepared = out_root / "prepared_data"
        _run_prepare_data(args.raw_data, prepared, args.valid_ratio)
        cfg.data.train_path = str(prepared / "train.jsonl")
        cfg.data.valid_path = str(prepared / "valid.jsonl")
    stages = [x.strip() for x in args.stages.split(",") if x.strip()]
    steps = _parse_steps(args.steps, stages, cfg.train.max_steps)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    resume = None
    stage_summary = []
    for stage, max_steps in zip(stages, steps):
        stage_cfg = deepcopy(cfg)
        stage_cfg.train.train_stage = stage
        stage_cfg.train.max_steps = max_steps
        assert_valid_training_config(stage_cfg)
        stage_dir = out_root / stage
        metrics = train_one_stage(stage_cfg, out_dir=stage_dir, device=device, resume=resume)
        stage_summary.append({"stage": stage, "out_dir": str(stage_dir), **metrics})
        resume = stage_dir / "last.pt"
    eval_path = out_root / "eval.json"
    _run_evaluate(resume, cfg.data.valid_path or cfg.data.train_path, eval_path, args.eval_batches, device)
    writeback = {"enabled": False, "count": 0}
    if args.writeback_format != "none":
        model, _ = load_checkpoint(resume, map_location="cpu")
        count = writeback_materialized_cold_units(model, out_root / "cold_units", fmt=args.writeback_format)
        writeback = {"enabled": True, "count": count, "format": args.writeback_format, "out_dir": str(out_root / "cold_units")}
    report = {
        "config": args.config,
        "out_dir": str(out_root),
        "stages": stage_summary,
        "eval": json.loads(eval_path.read_text(encoding="utf-8")),
        "writeback": writeback,
    }
    (out_root / "final_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _run_report(out_root / "final_report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _run_prepare_data(raw: str, out: Path, valid_ratio: float) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_data.py"),
        "--input",
        raw,
        "--out-dir",
        str(out),
        "--valid-ratio",
        str(valid_ratio),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _run_evaluate(checkpoint: Path, data: str | None, out: Path, max_batches: int, device: torch.device) -> None:
    if data is None:
        raise ValueError("evaluation data path is required")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        "--checkpoint",
        str(checkpoint),
        "--data",
        data,
        "--max-batches",
        str(max_batches),
        "--out",
        str(out),
        "--device",
        str(device),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _run_report(report: Path) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "report.py"), "--report", str(report)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _parse_steps(raw: str | None, stages: list[str], default: int) -> list[int]:
    if raw is None:
        return [default for _ in stages]
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(values) == 1:
        return values * len(stages)
    if len(values) != len(stages):
        raise ValueError("--steps must have one value or one value per stage")
    return values


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
