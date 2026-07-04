from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.training.runner import train_one_stage
from dopa_coder_n1.training.validation import assert_valid_training_config


DEFAULT_STAGES = ["stage1", "stage2", "stage3", "stage3_5"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal staged DOPA training schedule.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="runs/staged")
    parser.add_argument("--data", default=None)
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    parser.add_argument("--steps", default=None, help="Comma-separated max steps per stage.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume the first requested stage.")
    args = parser.parse_args()

    base_cfg = DOPAConfig.from_yaml(args.config)
    if args.data is not None:
        base_cfg.data.train_path = args.data
    stages = [x.strip() for x in args.stages.split(",") if x.strip()]
    steps = _parse_steps(args.steps, stages, base_cfg.train.max_steps)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    resume = args.resume
    summary = []
    for stage, max_steps in zip(stages, steps):
        cfg = deepcopy(base_cfg)
        cfg.train.train_stage = stage
        cfg.train.max_steps = max_steps
        assert_valid_training_config(cfg)
        stage_dir = out_root / stage
        metrics = train_one_stage(cfg, out_dir=stage_dir, device=device, resume=resume)
        summary.append({"stage": stage, "out_dir": str(stage_dir), **metrics})
        resume = stage_dir / "last.pt"
    (out_root / "stages_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


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
