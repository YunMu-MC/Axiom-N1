from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def function_to_skeleton(source: str) -> dict:
    tree = ast.parse(source)
    func = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if func is None:
        raise ValueError("source does not contain a top-level Python function")
    params = [arg.arg for arg in func.args.args]
    steps: list[dict[str, str]] = []
    if params:
        steps.append({"op": "parse_input"})
    has_return = False
    for node in ast.walk(func):
        op = _node_op(node)
        if op == "return":
            has_return = True
            continue
        if op is not None and {"op": op} not in steps:
            steps.append({"op": op})
    if has_return:
        steps.append({"op": "return"})
    elif not steps:
        steps.append({"op": "emit_output"})
    elif steps[-1]["op"] != "emit_output":
        steps.append({"op": "emit_output"})
    return {
        "kind": "python_function",
        "name": func.name,
        "params": params,
        "steps": steps,
    }


def iter_python_function_skeleton_records(
    paths: Iterable[Path],
    *,
    sample_rate: float,
    seed: int,
) -> Iterator[dict]:
    rng = random.Random(seed)
    for path in paths:
        for file in _iter_python_files(path):
            source = file.read_text(encoding="utf-8", errors="ignore")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if sample_rate < 1.0 and rng.random() > sample_rate:
                    continue
                segment = ast.get_source_segment(source, node) or _fallback_segment(source, node)
                skeleton = function_to_skeleton(segment)
                skeleton_json = json.dumps(skeleton, ensure_ascii=False, sort_keys=True)
                yield {
                    "text": f"[skeleton]\nCode:\n{segment.strip()}\nSkeleton JSON:\n{skeleton_json}",
                    "skeleton": skeleton,
                    "metadata": {
                        "source_path": str(file),
                        "function": skeleton["name"],
                        "task": "skeleton_generation",
                    },
                }


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate [skeleton] multitask records from Python functions.")
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    count = write_jsonl(
        args.out,
        iter_python_function_skeleton_records(args.input, sample_rate=args.sample_rate, seed=args.seed),
    )
    print(f"wrote={count} out={args.out}")
    return 0


def _iter_python_files(path: Path) -> Iterator[Path]:
    if path.is_file() and path.suffix == ".py":
        yield path
    elif path.is_dir():
        yield from path.rglob("*.py")


def _node_op(node: ast.AST) -> str | None:
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        return "loop"
    if isinstance(node, ast.If):
        return "branch"
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return "assign"
    if isinstance(node, ast.Call):
        return "call"
    if isinstance(node, ast.Return):
        return "return"
    return None


def _fallback_segment(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    start = max(0, getattr(node, "lineno", 1) - 1)
    end = getattr(node, "end_lineno", start + 1)
    return "\n".join(lines[start:end])


if __name__ == "__main__":
    raise SystemExit(main())
