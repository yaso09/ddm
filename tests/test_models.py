"""Model factory and parameter-counting helpers."""

from __future__ import annotations

import pytest
import torch

from ddm.models import DDMModel, build_model, count_parameters
from tests.conftest import make_config


def test_count_parameters_matches_module() -> None:
    """count_parameters equals sum over trainable parameters."""
    model = build_model(make_config("ddm"))
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count_parameters(model) == expected


def test_build_model_rejects_unknown_type() -> None:
    """build_model must raise ValueError for unknown model types."""
    with pytest.raises(ValueError):
        build_model(make_config("unknown"))


def test_build_model_ddm_ablation_never_learns_g() -> None:
    """build_model for ddm_ablation yields learn_g=False even if config says True."""
    model = build_model(make_config("ddm_ablation", learn_g=True))
    assert isinstance(model, DDMModel)
    assert not model.learn_g
    assert all(not block.attn.learn_g for block in model.blocks)


def test_ddmmodel_explicit_constructor() -> None:
    """DDMModel works with explicit arguments (no config)."""
    model = DDMModel(vocab_size=1000, d_model=32, n_layers=1, n_heads=4, max_seq_len=64)
    x = torch.randint(0, 1000, (1, 8))
    logits, memory = model(x)
    assert logits.shape == (1, 8, 1000)
    assert len(memory) == 1