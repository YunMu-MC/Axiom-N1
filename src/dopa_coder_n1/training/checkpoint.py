from __future__ import annotations

from pathlib import Path

import torch

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.fine_cold import ColdUnitId
from dopa_coder_n1.model.dopa import DOPACoderN1


def save_checkpoint(
    path: str | Path,
    model: DOPACoderN1,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    cfg: DOPAConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "config": cfg.to_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[DOPACoderN1, dict]:
    raw = torch.load(path, map_location=map_location)
    cfg = DOPAConfig.from_dict(raw["config"])
    model = DOPACoderN1(cfg)
    _materialize_lazy_units_for_state(model, raw["model"])
    state = _fill_missing_shadow_state(model, raw["model"])
    model.load_state_dict(state, strict=True)
    return model, raw


def _materialize_lazy_units_for_state(model: DOPACoderN1, state: dict) -> None:
    if not getattr(model, "use_fine_cold", False):
        return
    prefix = "fine_cold_shell.store.unit_"
    seen: set[str] = set()
    for key in state:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        unit_key = rest.split(".", 1)[0]
        if unit_key in seen:
            continue
        seen.add(unit_key)
        parts = unit_key.split("_")
        if len(parts) != 3:
            continue
        layer = int(parts[0][1:])
        kind = parts[1]
        index = int(parts[2])
        model.fine_cold_shell.store.load(ColdUnitId(layer=layer, kind=kind, index=index))


def _fill_missing_shadow_state(model: DOPACoderN1, state: dict) -> dict:
    current = model.state_dict()
    if all(key in state for key in current):
        return state
    patched = dict(state)
    for key, value in current.items():
        if key in patched:
            continue
        if (
            key.endswith(".delta")
            or key.endswith(".mask")
            or key.endswith(".shadow_scale")
            or key.endswith(".shadow_int8")
            or key.endswith(".packed_weight")
            or key.endswith(".weight_scale")
        ):
            patched[key] = value
    return {key: value for key, value in patched.items() if key in current}
