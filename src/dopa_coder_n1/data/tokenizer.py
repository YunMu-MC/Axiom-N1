from __future__ import annotations

import json
from pathlib import Path
import subprocess

from dopa_coder_n1.model.rust_core import RustCoreBackend


class ByteTokenizer:
    """UTF-8 byte tokenizer with stable special IDs for from-scratch training."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    vocab_size = 260

    def __init__(self, rust_backend: RustCoreBackend | None = None):
        self.rust_backend = rust_backend if rust_backend is not None else RustCoreBackend.default()
        self.backend_name = "python"

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if self.rust_backend is not None and self.rust_backend.available():
            try:
                ids = self.rust_backend.encode_bytes(text, add_bos=add_bos, add_eos=add_eos)
                self.backend_name = "rust"
                return ids
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
                self.backend_name = "python"
        ids = [b + 4 for b in text.encode("utf-8", errors="replace")]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        if self.rust_backend is not None and self.rust_backend.available():
            try:
                text = self.rust_backend.decode_bytes(ids)
                self.backend_name = "rust"
                return text
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
                self.backend_name = "python"
        raw = bytes([i - 4 for i in ids if i >= 4])
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "byte", "vocab_size": self.vocab_size}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        _ = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls()
