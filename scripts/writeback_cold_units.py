from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.training.checkpoint import load_checkpoint
from dopa_coder_n1.training.cold_units import writeback_materialized_cold_units


def main() -> None:
    parser = argparse.ArgumentParser(description="Write materialized fine-grained cold units to disk.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--format", choices=["int4", "int8", "fp32"], default="int4")
    args = parser.parse_args()
    model, _ = load_checkpoint(args.checkpoint, map_location="cpu")
    count = writeback_materialized_cold_units(model, args.out_dir, fmt=args.format)
    print(f"wrote {count} cold units to {args.out_dir} as {args.format}")


if __name__ == "__main__":
    main()
