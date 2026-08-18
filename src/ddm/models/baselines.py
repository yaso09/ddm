"""Baseline language models used for comparison against DDM."""

from __future__ import annotations

import torch
from torch import nn


class BigramModel(nn.Module):
    """First-order Markov baseline: ``P(x_t | x_{t-1})``.

    A token embedding followed by a linear projection; ``logits[:, t]`` is
    computed from ``x[:, t]`` alone, so the model is causal by construction.

    Args:
        vocab_size: Vocabulary size.
        d_model: Embedding dimension.
        dropout: Unused for this model (kept for API parity).
    """

    def __init__(self, vocab_size: int, d_model: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map token ids to logits.

        Args:
            x: Token ids ``[B, T]``.

        Returns:
            Logits ``[B, T, vocab_size]``.
        """
        return self.proj(self.emb(x))


class NGramModel(nn.Module):
    """Fixed-window n-gram baseline: ``P(x_t | x_{t-n+1}, ..., x_t)``.

    The last ``n_context`` tokens are embedded, concatenated and passed
    through a one-hidden-layer MLP.

    Args:
        vocab_size: Vocabulary size.
        d_model: Embedding/hidden dimension.
        n_context: Window size ``n``.
        dropout: Dropout applied on the concatenated embedding (0 = off).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_context: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_context = n_context
        self.emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * n_context, d_model),
            nn.ReLU(),
        )
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map token ids to logits.

        Args:
            x: Token ids ``[B, T]``.

        Returns:
            Logits ``[B, T, vocab_size]``. Early positions see padded zeros.
        """
        B, T = x.shape
        pad = torch.zeros(
            (B, self.n_context - 1), dtype=x.dtype, device=x.device
        )
        x_padded = torch.cat([pad, x], dim=1)
        windows = x_padded.unfold(1, self.n_context, 1)
        h = self.emb(windows)
        h = self.dropout(h.reshape(B, T, -1))
        return self.proj(self.mlp(h))


class SmallTransformerLM(nn.Module):
    """Reference decoder-only transformer with learned absolute positions.

    Built from standard ``nn.TransformerEncoderLayer`` blocks with a causal
    mask (no positional bias such as ALiBi or relative embeddings).

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden/embedding dimension.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length (learned position embeddings).
        dropout: Dropout probability (feed-forward + attention).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map token ids to logits.

        Args:
            x: Token ids ``[B, T]`` with ``T <= max_seq_len``.

        Returns:
            Logits ``[B, T, vocab_size]``.
        """
        _, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.tok_emb(x) + self.pos_emb(pos))
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        h = self.encoder(h, mask=causal_mask)
        return self.proj(self.ln_f(h))
