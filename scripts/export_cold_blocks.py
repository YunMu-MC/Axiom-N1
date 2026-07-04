from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.fine_cold import ColdUnitId
from dopa_coder_n1.model.quantization import quantize_int4_state_dict, quantize_int8_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cold shell blocks to disk for lazy loading.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--int4", action="store_true")
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()
    if args.int4 and args.int8:
        raise ValueError("choose only one cold-shell export format: --int4 or --int8")

    cfg = DOPAConfig.from_yaml(args.config)
    cfg.offload.lazy_cold_blocks = True
    model = DOPACoderN1(cfg)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if model.use_fine_cold:
        for layer in range(cfg.model.cold_layers):
            for head in range(cfg.model.n_heads):
                unit_id = ColdUnitId(layer=layer, kind="head", index=head)
                unit = model.fine_cold_shell.store.load(unit_id).to("cpu")
                state = unit.state_dict()
                torch.save(
                    _pack_state(state, int4=args.int4, int8=args.int8),
                    out / f"{unit_id.key}{_suffix(args.int4, args.int8)}.pt",
                )
            for block in range(cfg.offload.cold_ffn_subblocks):
                unit_id = ColdUnitId(layer=layer, kind="ffn", index=block)
                unit = model.fine_cold_shell.store.load(unit_id).to("cpu")
                state = unit.state_dict()
                torch.save(
                    _pack_state(state, int4=args.int4, int8=args.int8),
                    out / f"{unit_id.key}{_suffix(args.int4, args.int8)}.pt",
                )
            print(f"saved fine-grained layer {layer}")
    else:
        for idx in range(model.cold_manager.num_blocks):
            block = model.cold_manager.load_block_weights(idx).to("cpu")
            state = block.state_dict()
            torch.save(
                _pack_state(state, int4=args.int4, int8=args.int8),
                out / f"block_{idx}{_suffix(args.int4, args.int8)}.pt",
            )
            print(f"saved block {idx}")


def _pack_state(
    state: dict[str, torch.Tensor],
    *,
    int4: bool,
    int8: bool,
) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
    if int4:
        return quantize_int4_state_dict(state)
    if int8:
        return quantize_int8_state_dict(state)
    return state


def _suffix(int4: bool, int8: bool) -> str:
    if int4:
        return ".int4"
    if int8:
        return ".int8"
    return ""


if __name__ == "__main__":
    main()
