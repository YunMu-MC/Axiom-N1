from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

from dopa_coder_n1.model.rust_core import RustCoreBackend


class SkeletonTokenizer:
    """Deterministic byte/hash tokenizer for structured skeleton JSON."""

    def __init__(self, vocab_size: int, rust_backend: RustCoreBackend | None = None):
        self.vocab_size = vocab_size
        self.rust_backend = rust_backend if rust_backend is not None else RustCoreBackend.default()
        self.backend_name = "python"

    def encode(self, skeleton: dict | str, max_len: int = 256) -> list[int]:
        if isinstance(skeleton, dict):
            fast = self._encode_dict_with_rust(skeleton, max_len=max_len)
            if fast is not None:
                return fast
        text = json.dumps(skeleton, sort_keys=True) if isinstance(skeleton, dict) else skeleton
        raw = text.encode("utf-8")
        ids = [1]
        ids.extend((b % (self.vocab_size - 4)) + 4 for b in raw[: max_len - 2])
        ids.append(2)
        ids.extend([0] * (max_len - len(ids)))
        return ids[:max_len]

    def load_jsonl(self, path: str | Path, max_len: int = 256) -> list[list[int]]:
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(self.encode(json.loads(line), max_len=max_len))
        return rows

    def _encode_dict_with_rust(self, skeleton: dict, *, max_len: int) -> list[int] | None:
        if self.rust_backend is None or not self.rust_backend.available():
            self.backend_name = "python"
            return None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "skeleton.json"
                path.write_text(json.dumps(skeleton, sort_keys=True), encoding="utf-8")
                ids = self.rust_backend.encode_skeleton_json(path, vocab_size=self.vocab_size, max_len=max_len)
            self.backend_name = "rust"
            return ids
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
            self.backend_name = "python"
            return None
