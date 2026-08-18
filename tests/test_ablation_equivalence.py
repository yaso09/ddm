"""Ablation equivalence: with learn_g=False the gate is exactly 1/k."""

from __future__ import annotations

import torch

from ddm.models import DDMModel, build_model
from tests.conftest import make_config


def test_fixed_gate_is_one_over_k() -> None:
    """attn.g(k) must equal 1/k for k = 1..L when learn_g is False."""
    config = make_config("ddm_ablation")
    model = build_model(config)
    block = model.blocks[0]
    assert not block.attn.learn_g
    assert block.attn.g_mlp is None

    k = torch.arange(1, 65)
    g = block.attn.g(k)
    assert torch.allclose(g, 1.0 / k)


def test_ablation_matches_fixed_gate_forward() -> None:
    """The ablation equals a learned-gate model whose MLP is pinned to 1/k."""
    torch.manual_seed(10)
    ablation = build_model(make_config("ddm_ablation"))
    learned = build_model(make_config("ddm", learn_g=False))

    # Both models are functionally identical (same weights, same 1/k gate).
    ablation.load_state_dict(learned.state_dict())
    x = torch.randint(0, 1000, (2, 16))
    logits_a, _ = ablation(x)
    logits_b, _ = learned(x)
    assert torch.allclose(logits_a, logits_b, atol=1e-6)


def test_g_curve_rows_are_one_over_k() -> None:
    """get_g_curve must return the fixed 1/k curve for the ablation."""
    model = build_model(make_config("ddm_ablation"))
    k, curves = model.get_g_curve(max_k=32)
    expected = 1.0 / k.astype("float32")[None, :]
    assert torch.allclose(torch.as_tensor(curves), torch.as_tensor(expected), atol=1e-6)


def test_ablation_config_forces_learn_g_false() -> None:
    """DDMModel.from_config must never learn g for model_type ddm_ablation."""
    model = DDMModel.from_config(make_config("ddm_ablation", learn_g=True))
    assert not model.learn_g
    assert all(not block.attn.learn_g for block in model.blocks)