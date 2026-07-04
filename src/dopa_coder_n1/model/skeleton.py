from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dopa_coder_n1.data.skeleton_tokenizer import SkeletonTokenizer


@dataclass
class SkeletonBatch:
    token_ids: torch.Tensor
    adjacency: torch.Tensor | None = None


class SkeletonCompiler(nn.Module):
    """Transformer encoder for task skeletons.

    The paper describes a DAG/GIN compiler. This implementation accepts tokenized JSON
    and optional adjacency. The token path is complete and trainable; the adjacency hook is
    kept for graph-aware extensions without changing DOPA's interface.
    """

    def __init__(self, vocab_size: int, skeleton_dim: int, layers: int, heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, skeleton_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=skeleton_dim,
            nhead=heads,
            dim_feedforward=4 * skeleton_dim,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(skeleton_dim)

    def forward(self, batch: SkeletonBatch) -> torch.Tensor:
        mask = batch.token_ids.eq(0)
        x = self.embedding(batch.token_ids)
        x = self.encoder(x, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)
