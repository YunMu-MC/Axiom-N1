from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_ids(stdout: str) -> list[int]:
    text = stdout.strip()
    if not text:
        return []
    return [int(item) for item in text.split()]


@dataclass(slots=True)
class RustCoreBackend:
    binary: Path

    @classmethod
    def default(cls) -> "RustCoreBackend":
        root = _project_root()
        exe = "dopa_core.exe" if __import__("os").name == "nt" else "dopa_core"
        release = root / "rust" / "dopa_core" / "target" / "release" / exe
        debug = root / "rust" / "dopa_core" / "target" / "debug" / exe
        return cls(release if release.exists() else debug)

    def available(self) -> bool:
        if not self.binary.exists():
            return False
        try:
            result = subprocess.run(
                [str(self.binary), "health"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=5,
            )
        except OSError:
            return False
        return result.returncode == 0 and result.stdout.strip() == "ok"

    def encode_bytes(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        result = subprocess.run(
            [
                str(self.binary),
                "byte-encode",
                text,
                "1" if add_bos else "0",
                "1" if add_eos else "0",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        return _parse_ids(result.stdout)

    def decode_bytes(self, ids: Iterable[int]) -> str:
        joined = " ".join(str(int(item)) for item in ids)
        result = subprocess.run(
            [str(self.binary), "byte-decode", joined],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        return result.stdout

    def encode_byte_lines(
        self,
        path: str | Path,
        *,
        add_bos_first: bool = False,
        add_eos_each: bool = False,
    ) -> list[list[int]]:
        result = subprocess.run(
            [
                str(self.binary),
                "byte-encode-lines",
                str(path),
                "1" if add_bos_first else "0",
                "1" if add_eos_each else "0",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        return [json.loads(line) for line in result.stdout.splitlines()]

    def encode_skeleton_json(self, path: str | Path, *, vocab_size: int, max_len: int) -> list[int]:
        result = subprocess.run(
            [
                str(self.binary),
                "skeleton-encode-json",
                str(path),
                str(int(vocab_size)),
                str(int(max_len)),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        return _parse_ids(result.stdout)

    def rank_texts(self, *, query: str, rows: Iterable[dict], limit: int) -> list[dict]:
        if not self.available():
            raise RuntimeError(f"Rust retrieval backend is not available: {self.binary}")
        clean_rows = []
        for row in rows:
            row_id = str(row["id"])
            text = str(row.get("text", "")).replace("\t", " ").replace("\r", " ").replace("\n", " ")
            clean_rows.append((row_id, text))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank_input.tsv"
            path.write_text(
                "".join(f"{row_id}\t{text}\n" for row_id, text in clean_rows),
                encoding="utf-8",
            )
            result = subprocess.run(
                [str(self.binary), "rank-text", query, str(path), str(max(0, int(limit)))],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
            )
        ranked = []
        for line in result.stdout.splitlines():
            row_id, score, text = (line.split("\t", 2) + ["", ""])[:3]
            ranked.append({"id": row_id, "score": float(score), "text": text})
        return ranked


RustRetrievalBackend = RustCoreBackend
