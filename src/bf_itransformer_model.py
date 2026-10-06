from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible instance normalization over the temporal axis."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x = (x - mean) / std
        return x * self.gamma + self.beta


class InvertedTransformerBlock(nn.Module):
    """Variable-token self-attention block."""

    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.dropout1(y)
        x = x + self.ffn(self.norm2(x))
        return x


class BFiTransformerLite(nn.Module):
    """Lightweight iTransformer-style model for blast furnace Si residual prediction."""

    def __init__(
        self,
        window: int,
        channels: int,
        d_model: int = 32,
        num_heads: int = 4,
        ffn_hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.use_revin = use_revin
        self.revin = RevIN(channels) if use_revin else nn.Identity()
        self.token_proj = nn.Linear(window, d_model)
        self.blocks = nn.ModuleList(
            [
                InvertedTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_hidden=ffn_hidden,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model * 2, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        seq = x[:, 0]
        seq = self.revin(seq)
        tokens = seq.transpose(1, 2)
        tokens = self.token_proj(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        pooled_mean = tokens.mean(dim=1)
        pooled_max = tokens.max(dim=1).values
        return torch.cat([pooled_mean, pooled_max], dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(x)
        return {"si": self.residual_head(hidden).squeeze(-1)}
