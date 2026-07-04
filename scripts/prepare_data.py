from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/valid files for DOPA training.")
    parser.add_argument("--input", required=True, help="Input file or directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    lines = list(_iter_lines(Path(args.input)))
    rng.shuffle(lines)
    valid_count = max(1, int(len(lines) * args.valid_ratio)) if lines else 0
    valid = lines[:valid_count]
    train = lines[valid_count:]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("".join(train), encoding="utf-8")
    (out / "valid.jsonl").write_text("".join(valid), encoding="utf-8")
    print(f"train={len(train)} valid={len(valid)} out={out}")


def _iter_lines(path: Path):
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                yield line + "\n"
        return
    for file in path.rglob("*"):
        if file.suffix.lower() not in {".txt", ".py", ".md", ".jsonl"}:
            continue
        text = file.read_text(encoding="utf-8", errors="ignore")
        if file.suffix.lower() == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    yield line + "\n"
        elif text.strip():
            escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            yield f'{{"text": "{escaped}"}}\n'


if __name__ == "__main__":
    main()
