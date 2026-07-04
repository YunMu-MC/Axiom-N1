from __future__ import annotations

from pathlib import Path

import torch

from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.quantization import quantize_int4_state_dict, quantize_int8_state_dict


def writeback_materialized_cold_units(
    model: DOPACoderN1,
    out_dir: str | Path,
    *,
    fmt: str = "int4",
) -> int:
    if not getattr(model, "use_fine_cold", False):
        return 0
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for key, unit in model.fine_cold_shell.store._cache.items():
        state = {name: tensor.detach().cpu() for name, tensor in unit.state_dict().items()}
        if fmt == "int4":
            payload = quantize_int4_state_dict(state)
            suffix = ".int4.pt"
        elif fmt == "int8":
            payload = quantize_int8_state_dict(state)
            suffix = ".int8.pt"
        elif fmt == "fp32":
            payload = state
            suffix = ".pt"
        else:
            raise ValueError("fmt must be one of: int4, int8, fp32")
        torch.save(payload, out / f"{key}{suffix}")
        count += 1
    return count
