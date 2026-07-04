from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.utils.memory import estimate_memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate DOPA memory against local hardware budgets.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = DOPAConfig.from_yaml(args.config)
    report = estimate_memory(cfg)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
