"""Residual block wrapping distance-aware attention and a feed-forward network."""

from __future__ import annotations

import torch
from torch import nn

from ddm.models.attention import DistanceAwareAttention


class DistanceAwareBlock(nn.Module):
    """Pre-LayerNorm residual block with a distance-aware attention and FFN.

    Following the pre-LN Transformer convention, layer norms are applied
    inside the residual branches:

    ``x = x + dropout(attn(ln1(x), memory))``
    ``x = x + dropout(ffn(ln2(x)))``

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        learn_g: Passed to :class:`DistanceAwareAttention`.
        max_seq_len: Passed to :class:`DistanceAwareAttention` (distance
            normalization).
        dropout: Dropout probability applied to both residual branches.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        learn_g: bool = True,
        max_seq_len: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = DistanceAwareAttention(
            d_model=d_model,
            n_heads=n_heads,
            learn_g=learn_g,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Input hidden states ``[B, T, d_model]``.
            memory: Optional segment memory vector ``[B, d_model]``.

        Returns:
            Output hidden states ``[B, T, d_model]``.
        """
        x = x + self.dropout(self.attn(self.ln1(x), memory=memory))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
