from __future__ import annotations

import torch
from torch import nn


class PatchTSTRegressor(nn.Module):
    """Compact channel-independent PatchTST regressor."""

    def __init__(
        self,
        channels: int,
        window: int,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if window < patch_len:
            raise ValueError("window must be at least patch_len")
        self.channels = channels
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = 1 + (window - patch_len) // stride
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.channel_gate = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(16, d_model // 2), 1),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, channels]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        x = ((x - mean) / std).transpose(1, 2)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        batch, channels, n_patches, _ = patches.shape
        tokens = self.patch_embed(patches.reshape(batch * channels, n_patches, -1))
        tokens = self.encoder(tokens + self.position[:, :n_patches])
        channel_embeddings = self.norm(tokens).mean(dim=1).reshape(batch, channels, -1)
        weights = torch.softmax(self.channel_gate(channel_embeddings), dim=1)
        weighted = (weights * channel_embeddings).sum(dim=1)
        maximum = channel_embeddings.max(dim=1).values
        return self.head(torch.cat([weighted, maximum], dim=-1)).squeeze(-1)
