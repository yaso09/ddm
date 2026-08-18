"""The full Distance-Decomposed Model (DDM) language model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ddm.config import DDMConfig
from ddm.models.blocks import DistanceAwareBlock


class DDMModel(nn.Module):
    """Distance-Decomposed Model: a decoder-only LM with distance-aware blocks.

    ``P(x_1..x_n) = prod_t P(x_t | x_<t)`` is kept in full (no Markov
    truncation); the conditional probabilities are computed by attention
    layers that decompose the influence of the past by distance (ALiBi
    penalty + learned gate g(k), see :class:`DistanceAwareAttention`).

    **Segment memory:** after every block the mean of its hidden states is
    stored (detached) as a per-layer memory vector. The caller passes these
    vectors back as ``memory`` on the next call; each one is prepended to the
    keys/values as a single extra token that the causal mask always keeps
    visible. This gives the model indirect access to an arbitrarily long
    past at constant cost (Transformer-XL-inspired, simplified).

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden/embedding dimension.
        n_layers: Number of distance-aware blocks.
        n_heads: Number of attention heads per block.
        max_seq_len: Maximum sequence length.
        learn_g: Whether the distance gate is learned (False = fixed 1/k).
        dropout: Dropout probability used in blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 128,
        learn_g: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DistanceAwareBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    learn_g=learn_g,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.learn_g = learn_g

    @classmethod
    def from_config(cls, config: DDMConfig) -> DDMModel:
        """Build a model from a :class:`DDMConfig`.

        Args:
            config: Model configuration (``model_type`` ``ddm`` or
                ``ddm_ablation``).

        Returns:
            The constructed model.
        """
        learn_g = config.learn_g and config.model_type == "ddm"
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            learn_g=learn_g,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the model over a batch of token sequences.

        Args:
            x: Token ids ``[B, T]`` (the model is causal: ``logits[:, t]``
                only depends on ``x[:, :t]`` and predicts ``x[:, t + 1]``).
            memory: Optional list of per-layer memory vectors ``[B, d_model]``
                carried over from the previous call; each is appended as an
                always-visible key/value token.

        Returns:
            Tuple ``(logits, new_memory)``: logits ``[B, T, vocab_size]`` and
            the per-layer detached segment memories to pass to the next call.
        """
        h = self.dropout(self.tok_emb(x))
        new_memory: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            mem_i = memory[i] if memory is not None else None
            h = block(h, memory=mem_i)
            new_memory.append(h.mean(dim=1).detach())
        h = self.ln_f(h)
        logits = self.proj(h)
        return logits, new_memory

    def get_g_curve(
        self,
        max_k: int = 64,
        device: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the distance gate g(k) on ``k = 1..max_k`` for every layer.

        Args:
            max_k: Largest distance to evaluate.
            device: Torch device on which to evaluate; empty means the
                device of the model's parameters.

        Returns:
            Tuple ``(k, curves)`` where ``k`` is ``[max_k]`` and ``curves``
            is ``[n_layers, max_k]`` (for the ablation the rows are the fixed
            ``1/k`` curve).
        """
        if not device:
            device = str(next(self.parameters()).device)
        k = torch.arange(1, max_k + 1, device=device)
        curves = []
        with torch.no_grad():
            for block in self.blocks:
                curves.append(block.attn.g(k))
        return k.cpu().numpy(), torch.stack(curves).cpu().numpy()
