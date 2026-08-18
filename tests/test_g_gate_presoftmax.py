"""Regression test: the distance gate g(k) is applied in log-space
BEFORE the softmax, never as a post-softmax multiplicative re-weighting.

Rationale: an earlier experiment applied g(k) after the softmax, which
degenerated into a correction term that merely restored the probability
mass ALiBi had suppressed instead of learning an independent distance
signal. This test locks the pre-softmax behavior: the raw scores must
contain the additive ``log g(k)`` term.
"""

from __future__ import annotations

import math

import torch

from ddm.models import DistanceAwareAttention

D = 16
H = 2
MAX_LEN = 32
T = 4


def _fixed_attention() -> DistanceAwareAttention:
    """Attention with identity Q/K/V/out projections and a known g(k)."""
    attn = DistanceAwareAttention(
        d_model=D, n_heads=H, learn_g=True, max_seq_len=MAX_LEN, dropout=0.0
    )
    with torch.no_grad():
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
            proj.weight.copy_(torch.eye(D))
            proj.bias.zero_()
        # g(k) = sigmoid(ReLU(k / MAX_LEN) + 0) = sigmoid(k / MAX_LEN):
        # first layer maps k_norm into 16 identical ReLU units, the second
        # averages them.
        attn.g_mlp[0].weight.copy_(torch.ones(16, 1))
        attn.g_mlp[0].bias.zero_()
        attn.g_mlp[2].weight.copy_(torch.ones(1, 16) / 16)
        attn.g_mlp[2].bias.zero_()
    return attn


def _expected_gate(distance: torch.Tensor) -> torch.Tensor:
    """Reference g(k) = sigmoid(k / MAX_LEN) for the fixed MLP above."""
    return torch.sigmoid(distance.float() / MAX_LEN)


def test_gate_applied_presoftmax_in_log_space() -> None:
    """Scores must equal raw + ALiBi + log g(k) + causal mask."""
    torch.manual_seed(3)
    attn = _fixed_attention().eval()
    x = torch.randn(1, T, D)

    with torch.no_grad():
        scores, attn_actual = attn.get_attention_weights(x)

    q_idx = torch.arange(T).view(T, 1)
    key_pos = torch.arange(T).view(1, T)
    distance = (q_idx - key_pos).clamp(min=0).float()
    causal = torch.where(key_pos <= q_idx, 0.0, float("-inf"))

    # scores are per-head: split x into heads before the dot product
    hd = D // H
    xh = x.view(1, T, H, hd).transpose(1, 2)
    raw = torch.matmul(xh, xh.transpose(-2, -1)) / math.sqrt(hd)
    alibi = -attn.alibi_slopes.view(1, H, 1, 1) * distance.view(1, 1, T, T)
    log_gate = torch.log(_expected_gate(distance).clamp(min=1e-6)).view(1, 1, T, T)
    expected_scores = raw + alibi + log_gate + causal

    assert scores.shape == (1, H, T, T)
    assert torch.allclose(scores, expected_scores, atol=1e-6, rtol=1e-5), (
        "scores must contain the additive log g(k) BEFORE the softmax"
    )
    assert torch.allclose(attn_actual, torch.softmax(expected_scores, dim=-1), atol=1e-6, rtol=1e-5)


def test_gate_is_not_postsoftmax_multiplication() -> None:
    """The attention must NOT equal g(k) * softmax(raw + alibi) without renormalization."""
    torch.manual_seed(4)
    attn = _fixed_attention().eval()
    x = torch.randn(1, T, D)

    with torch.no_grad():
        _, attn_actual = attn.get_attention_weights(x)

    q_idx = torch.arange(T).view(T, 1)
    key_pos = torch.arange(T).view(1, T)
    distance = (q_idx - key_pos).clamp(min=0).float()
    causal = torch.where(key_pos <= q_idx, 0.0, float("-inf"))
    hd = D // H
    xh = x.view(1, T, H, hd).transpose(1, 2)
    raw = torch.matmul(xh, xh.transpose(-2, -1)) / math.sqrt(hd)
    alibi = -attn.alibi_slopes.view(1, H, 1, 1) * distance.view(1, 1, T, T)

    attn_post = _expected_gate(distance) * torch.softmax(raw + alibi + causal, dim=-1)
    assert not torch.allclose(attn_actual, attn_post, atol=1e-4), (
        "gate must not be applied as post-softmax multiplication"
    )


def test_gate_range_and_monotonicity() -> None:
    """g(k) is sigmoid-bounded in (0, 1); the fixed test gate is monotone."""
    attn = _fixed_attention()
    k = torch.arange(0, MAX_LEN + 1).float()
    g = attn.g(k)
    assert (g > 0).all() and (g < 1).all()
    assert torch.allclose(g, _expected_gate(k), atol=1e-6)
