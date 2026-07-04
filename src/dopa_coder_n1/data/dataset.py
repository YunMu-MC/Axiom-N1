from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from dopa_coder_n1.data.skeleton_tokenizer import SkeletonTokenizer
from dopa_coder_n1.data.tokenizer import ByteTokenizer
from dopa_coder_n1.model.skeleton import SkeletonBatch


class PackedTextDataset(IterableDataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteTokenizer,
        seq_len: int,
        skeleton_tokenizer: SkeletonTokenizer | None = None,
        skeleton_len: int = 256,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.skeleton_tokenizer = skeleton_tokenizer
        self.skeleton_len = skeleton_len

    def _iter_files(self) -> Iterator[Path]:
        if self.path.is_file():
            yield self.path
        else:
            for suffix in ("*.txt", "*.py", "*.md", "*.jsonl"):
                yield from self.path.rglob(suffix)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | SkeletonBatch]]:
        buffer: list[int] = []
        token_buffers = {
            "external_knowledge_labels": [],
            "teacher_token_losses": [],
            "hot_token_losses": [],
        }
        for file in self._iter_files():
            for line in file.read_text(encoding="utf-8", errors="ignore").splitlines():
                skeleton = None
                text = line
                tool_need_label = None
                tool_argument_valid_label = None
                tool_query_target = None
                obj = {}
                if file.suffix == ".jsonl":
                    try:
                        obj = json.loads(line)
                        text = obj.get("text") or obj.get("solution") or ""
                        skeleton_obj = obj.get("skeleton")
                        if skeleton_obj is not None and self.skeleton_tokenizer is not None:
                            skeleton = torch.tensor(
                                self.skeleton_tokenizer.encode(skeleton_obj, self.skeleton_len),
                                dtype=torch.long,
                            )
                        tool_need_label = obj.get("tool_need_label")
                        tool_argument_valid_label = obj.get("tool_argument_valid_label")
                        tool_query_target = obj.get("tool_query_target")
                    except json.JSONDecodeError:
                        text = line
                        obj = {}
                encoded = self.tokenizer.encode(text + "\n", add_bos=not buffer)
                buffer.extend(encoded)
                _extend_token_supervision_buffers(token_buffers, obj, len(encoded))
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[: self.seq_len + 1]
                    buffer = buffer[self.seq_len :]
                    supervision_chunk = {
                        key: values[: self.seq_len]
                        for key, values in token_buffers.items()
                    }
                    for values in token_buffers.values():
                        del values[: self.seq_len]
                    ids = torch.tensor(chunk, dtype=torch.long)
                    item: dict[str, torch.Tensor] = {
                        "input_ids": ids[:-1],
                        "labels": ids[:-1].clone(),
                    }
                    if skeleton is None and self.skeleton_tokenizer is not None:
                        skeleton = torch.zeros(self.skeleton_len, dtype=torch.long)
                    if skeleton is not None:
                        item["skeleton_ids"] = skeleton
                    if _has_non_fill(supervision_chunk["external_knowledge_labels"], -1.0):
                        item["external_knowledge_labels"] = torch.tensor(
                            supervision_chunk["external_knowledge_labels"],
                            dtype=torch.float32,
                        )
                    if _has_finite(supervision_chunk["teacher_token_losses"]):
                        item["teacher_token_losses"] = torch.tensor(
                            supervision_chunk["teacher_token_losses"],
                            dtype=torch.float32,
                        )
                    if _has_finite(supervision_chunk["hot_token_losses"]):
                        item["hot_token_losses"] = torch.tensor(
                            supervision_chunk["hot_token_losses"],
                            dtype=torch.float32,
                        )
                    if tool_need_label is not None:
                        item["tool_need_labels"] = torch.tensor(int(tool_need_label), dtype=torch.long)
                    if tool_argument_valid_label is not None:
                        item["tool_argument_valid_labels"] = torch.tensor(
                            float(tool_argument_valid_label),
                            dtype=torch.float32,
                        )
                    if tool_query_target is not None:
                        item["tool_query_targets"] = torch.tensor(tool_query_target, dtype=torch.float32)
                    yield item


def collate_batch(items: list[dict]) -> dict:
    input_ids = torch.stack([x["input_ids"] for x in items])
    labels = torch.stack([x["labels"] for x in items])
    batch = {"input_ids": input_ids, "labels": labels}
    skeletons = [x.get("skeleton_ids") for x in items]
    if all(s is not None for s in skeletons):
        batch["skeleton"] = SkeletonBatch(token_ids=torch.stack(skeletons, dim=0))
    tool_need = [x.get("tool_need_labels") for x in items]
    if all(x is not None for x in tool_need):
        batch["tool_need_labels"] = torch.stack(tool_need, dim=0)
    tool_valid = [x.get("tool_argument_valid_labels") for x in items]
    if all(x is not None for x in tool_valid):
        batch["tool_argument_valid_labels"] = torch.stack(tool_valid, dim=0)
    tool_query = [x.get("tool_query_targets") for x in items]
    if all(x is not None for x in tool_query):
        batch["tool_query_targets"] = torch.stack(tool_query, dim=0)
    external_labels = _stack_optional_token_field(items, "external_knowledge_labels", fill_value=-1.0)
    if external_labels is not None:
        batch["external_knowledge_labels"] = external_labels
    teacher_losses = _stack_optional_token_field(items, "teacher_token_losses", fill_value=float("nan"))
    if teacher_losses is not None:
        batch["teacher_token_losses"] = teacher_losses
    hot_losses = _stack_optional_token_field(items, "hot_token_losses", fill_value=float("nan"))
    if hot_losses is not None:
        batch["hot_token_losses"] = hot_losses
    return batch


def _stack_optional_token_field(items: list[dict], key: str, *, fill_value: float) -> torch.Tensor | None:
    values = [item.get(key) for item in items]
    present = [value for value in values if value is not None]
    if not present:
        return None
    width = int(items[0]["input_ids"].numel())
    rows: list[torch.Tensor] = []
    for value in values:
        if value is None:
            rows.append(torch.full((width,), fill_value, dtype=torch.float32))
        else:
            tensor = value.to(dtype=torch.float32)
            if tensor.numel() != width:
                raise ValueError(f"{key} width mismatch: expected {width}, got {tensor.numel()}")
            rows.append(tensor)
    return torch.stack(rows, dim=0)


def _extend_token_supervision_buffers(buffers: dict[str, list[float]], obj: dict, encoded_len: int) -> None:
    buffers["external_knowledge_labels"].extend(
        _token_values(obj.get("external_knowledge_labels"), encoded_len, fill_value=-1.0)
    )
    buffers["teacher_token_losses"].extend(
        _token_values(obj.get("teacher_token_losses"), encoded_len, fill_value=float("nan"))
    )
    buffers["hot_token_losses"].extend(
        _token_values(obj.get("hot_token_losses"), encoded_len, fill_value=float("nan"))
    )


def _token_values(raw: object, length: int, *, fill_value: float) -> list[float]:
    if not isinstance(raw, list):
        return [fill_value] * length
    values: list[float] = []
    for item in raw:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            values.append(fill_value)
    if len(values) == length - 1:
        values.append(fill_value)
    if len(values) < length:
        values.extend([fill_value] * (length - len(values)))
    return values[:length]


def _has_non_fill(values: list[float], fill_value: float) -> bool:
    return any(value != fill_value for value in values)


def _has_finite(values: list[float]) -> bool:
    return any(value == value and value not in {float("inf"), -float("inf")} for value in values)
