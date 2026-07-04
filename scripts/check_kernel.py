from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.attention_backend import build_attention_backend
from dopa_coder_n1.model.kv_cache import LayerKVCache, pack_layer_kv, unpack_layer_kv
from dopa_coder_n1.model.triton_int4 import triton_int4_status, unpack_layer_kv_triton


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the final triton_int4 packed-KV backend.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "tiny_unit.yaml"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--require-kernel", action="store_true")
    args = parser.parse_args()

    cfg = DOPAConfig.from_yaml(args.config)
    if cfg.model.attention_backend != "triton_int4":
        raise SystemExit(f"config does not use final backend: {cfg.model.attention_backend}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but this PyTorch build cannot use CUDA")

    status = triton_int4_status(device)
    if args.require_kernel and not status["usable"]:
        raise SystemExit("triton_int4 kernel is not usable in this environment")

    backend = build_attention_backend(cfg.model.attention_backend)
    q = torch.randn(1, 2, 1, 4, device=device)
    k_new = torch.randn(1, 2, 1, 4, device=device)
    v_new = torch.randn(1, 2, 1, 4, device=device)
    cache = LayerKVCache(
        k=torch.randn(1, 2, 3, 4, device=device),
        v=torch.randn(1, 2, 3, 4, device=device),
    )
    packed = pack_layer_kv(cache)
    y, new_cache = backend.attention(
        q,
        k_new,
        v_new,
        kv_cache=packed,
        n_heads=2,
        n_kv_heads=2,
        dropout_p=0.0,
        is_training=False,
    )

    max_dequant_error = None
    if status["usable"]:
        ref = unpack_layer_kv(packed, device=device, dtype=q.dtype)
        got = unpack_layer_kv_triton(packed, device=device, dtype=q.dtype, allow_fallback=False)
        max_dequant_error = max(
            float((got.k - ref.k).abs().max().detach().cpu()),
            float((got.v - ref.v).abs().max().detach().cpu()),
        )

    report = {
        "config": str(Path(args.config).resolve()),
        "backend": cfg.model.attention_backend,
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "triton_installed": status["triton_installed"],
        "triton_int4_usable": status["usable"],
        "fallback_active": not status["usable"],
        "attention_shape": list(y.shape),
        "new_cache_shape": [list(new_cache[0].shape), list(new_cache[1].shape)],
        "max_dequant_error": max_dequant_error,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
