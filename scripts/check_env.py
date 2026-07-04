from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    print(f"python={sys.version}")
    print(f"executable={sys.executable}")
    for name in ["torch", "numpy", "yaml", "tqdm", "pytest"]:
        spec = importlib.util.find_spec(name)
        print(f"{name}={'OK' if spec else 'MISSING'}")
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        import torch

        from dopa_coder_n1.model.triton_int4 import triton_int4_status

        print(f"torch_version={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device={torch.cuda.get_device_name(0)}")
        status = triton_int4_status("cuda" if torch.cuda.is_available() else "cpu")
        print(f"triton_installed={status['triton_installed']}")
        print(f"triton_int4_usable={status['usable']}")
        if not status["usable"]:
            print("attention_backend=triton_int4 will use the portable int4 reference fallback")
    print(f"project={ROOT}")


if __name__ == "__main__":
    main()
