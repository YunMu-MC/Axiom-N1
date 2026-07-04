from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.dopa import DOPACoderN1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = DOPAConfig.from_yaml(args.config)
    model = DOPACoderN1(cfg)
    for k, v in model.parameter_report().items():
        print(f"{k}: {v:,}")


if __name__ == "__main__":
    main()
