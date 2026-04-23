"""JEPA encoder + predictor on per-layer residual-stream trajectories.

Input shape: (B, L, d_in)  — B trajectories, L layers, d_in = target-model hidden dim.
Encoder: Linear(d_in -> d_model) + learned per-layer positional embedding + TransformerEncoder.
Predictor: 2-layer MLP that maps encoder output at masked positions to target-encoder latents.
Target encoder: EMA copy of encoder, updated outside this module (see phase2/train.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerJEPA(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_L: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.max_L = max_L

        self.in_proj = nn.Linear(d_in, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(max_L, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full-trajectory encoding used by the target branch (no masking)."""
        B, L, _ = x.shape
        h = self.in_proj(x) + self.pos_emb[:L]
        return self.encoder(h)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Context encoding + prediction at masked layer positions.

        mask: (B, L) bool. True = layer is masked and must be predicted.
        Returns: (B, L, d_model) predictor outputs; caller selects positions via mask.
        """
        B, L, _ = x.shape
        h = self.in_proj(x) + self.pos_emb[:L]
        h = torch.where(mask.unsqueeze(-1), self.mask_token + self.pos_emb[:L], h)
        z = self.encoder(h)
        return self.predictor(z)


def random_block_mask(batch: int, L: int, ratio: float = 0.4, device: str = "cpu") -> torch.Tensor:
    """Contiguous-block mask of ~ratio fraction of L layers per example."""
    block_len = max(1, int(L * ratio))
    mask = torch.zeros(batch, L, dtype=torch.bool, device=device)
    starts = torch.randint(0, L - block_len + 1, (batch,), device=device)
    for b in range(batch):
        mask[b, starts[b]:starts[b] + block_len] = True
    return mask
