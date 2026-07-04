from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.training.runner import train_one_stage
from dopa_coder_n1.training.validation import assert_valid_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DOPA Coder N1 from scratch.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="runs/default")
    parser.add_argument("--data", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = DOPAConfig.from_yaml(args.config)
    if args.data is not None:
        cfg.data.train_path = args.data
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if cfg.data.train_path is None:
        raise SystemExit("Set data.train_path in config or pass --data")
    assert_valid_training_config(cfg)

    torch.manual_seed(cfg.train.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg.offload.device = str(device)
    metrics = train_one_stage(cfg, out_dir=args.out_dir, device=device, resume=args.resume)
    print(metrics)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
