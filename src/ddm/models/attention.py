"""Distance-aware attention for the Distance-Decomposed Model (DDM)."""

from __future__ import annotations

import math

import torch
from torch import nn

MIN_GATE_LOG: float = 1e-6


class DistanceAwareAttention(nn.Module):
    """Attention that decomposes context by the distance of each key token.

    A single attention layer combines three mechanisms:

    1. **ALiBi-style fixed distance penalty** -- per-head slope ``m_h`` is
       subtracted from the raw scores as ``-m_h * d(q, k)`` (Press et al.,
       2021). Slopes follow the standard geometric schedule
       ``2 ** (-8 * h / n_heads)``.
    2. **Learned distance gate g(k)** -- the distance, normalized to
       ``[0, 1]``, is passed through a small MLP
       (``Linear(1, 16) -> ReLU -> Linear(16, 1) -> sigmoid``) producing a
       positive weight in ``(0, 1)``. The gate is combined with the scores
       in log-space **before** the softmax
       (``scores + alibi + log g(k) + causal_mask``) and never as a
       post-softmax multiplicative re-weighting: applied after the softmax,
       g(k) degenerates into a correction term that merely restores the
       probability mass ALiBi suppressed instead of learning an independent
       signal (this failure mode is locked by ``tests/test_g_gate_presoftmax.py``).
    3. **Low-rank pairwise interaction** -- the term ``F(e_i, e_j) =
       (U e_i)^T (V e_j)`` from the theory document is mathematically
       equivalent to the standard query-key dot product, so it requires no
       extra module: the Q/K projections below *are* that term (a formal
       derivation lives in ``notebooks/01_Theory.ipynb`` and ``paper/``).

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads (ALiBi slope per head).
        learn_g: If True, g(k) is learned; if False it is frozen to ``1/k``
            (ablation, see ``test_ablation_equivalence.py``).
        max_seq_len: Maximum sequence length; used to normalize distances.
        dropout: Dropout applied to the attention weights.
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
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.learn_g = learn_g
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        slopes = 2.0 ** (-8.0 * torch.arange(1, n_heads + 1).float() / n_heads)
        self.register_buffer("alibi_slopes", slopes)

        if learn_g:
            self.g_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
        else:
            self.g_mlp = None

    def g(self, distance: torch.Tensor) -> torch.Tensor:
        """Compute the distance gate for the given distances.

        Args:
            distance: Tensor of non-negative integer distances between query
                and key positions.

        Returns:
            learn_g=True: ``sigmoid(MLP(distance / max_seq_len))`` in
            ``(0, 1)``, same shape as ``distance``. learn_g=False: the fixed
            curve ``1 / max(distance, 1)`` (ablation).
        """
        if self.learn_g:
            k_norm = (distance.float() / self.max_seq_len).unsqueeze(-1)
            return torch.sigmoid(self.g_mlp(k_norm)).squeeze(-1)
        return 1.0 / torch.clamp(distance.float(), min=1.0)

    def _attend(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute scores, attention weights and values.

        Args:
            x: Input hidden states of shape ``[B, T, d_model]``.
            memory: Optional per-layer segment memory ``[B, d_model]``; when
                provided it is detached and prepended to the keys/values as a
                single extra token at virtual position ``-1`` (always visible
                through the causal mask).

        Returns:
            Tuple ``(scores, attn, v)`` where ``scores`` are the pre-softmax
            scores ``[B, H, T, Tk]`` (with ALiBi, log g(k) and the causal
            mask already applied), ``attn`` the softmax weights of the same
            shape and ``v`` the projected values ``[B, H, Tk, head_dim]``.
        """
        B, T, _ = x.shape
        H, hd = self.n_heads, self.head_dim
        device = x.device

        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)

        if memory is not None:
            memory = memory.detach()
            kv_input = torch.cat([memory.unsqueeze(1), x], dim=1)
            key_pos = torch.cat(
                [
                    torch.full((1,), -1, device=device, dtype=torch.long),
                    torch.arange(T, device=device),
                ]
            )
        else:
            kv_input = x
            key_pos = torch.arange(T, device=device)

        Tk = kv_input.shape[1]
        k = self.k_proj(kv_input).view(B, Tk, H, hd).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, Tk, H, hd).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)

        q_idx = torch.arange(T, device=device).view(T, 1)
        distance = (q_idx - key_pos.view(1, Tk)).clamp(min=0).float()

        allow = (key_pos.view(1, Tk) <= q_idx) | (key_pos.view(1, Tk) < 0)
        causal_mask = torch.where(
            allow, torch.zeros((), device=device), torch.full((), float("-inf"), device=device)
        )

        alibi_bias = -self.alibi_slopes.view(1, H, 1, 1) * distance.view(1, 1, T, Tk)
        gate = self.g(distance)
        log_gate = torch.log(gate.clamp(min=MIN_GATE_LOG)).view(1, 1, T, Tk)

        scores = scores + alibi_bias + log_gate + causal_mask.view(1, 1, T, Tk)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        return scores, attn, v

    def get_attention_weights(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pre-softmax scores and the attention distribution.

        Args:
            x: Input hidden states ``[B, T, d_model]``.
            memory: Optional segment memory vector ``[B, d_model]``.

        Returns:
            Tuple ``(scores, attn)`` of shape ``[B, H, T, Tk]``; ``scores``
            include ALiBi, ``log g(k)`` and the causal mask.
        """
        scores, attn, _ = self._attend(x, memory)
        return scores, attn

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply distance-aware attention.

        Args:
            x: Input hidden states ``[B, T, d_model]``.
            memory: Optional segment memory vector ``[B, d_model]``.

        Returns:
            Contextualized output ``[B, T, d_model]``.
        """
        B, T, _ = x.shape
        _, attn, v = self._attend(x, memory)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)
