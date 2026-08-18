"""Forward output shapes for every model."""

from __future__ import annotations

import pytest
import torch

from ddm.models import DDMModel, build_model
from tests.conftest import BATCH, SEQ, VOCAB, make_config

ALL_TYPES = ["ddm", "ddm_ablation", "bigram", "ngram", "transformer"]


@pytest.mark.parametrize("model_type", ALL_TYPES)
def test_forward_shape(model_type: str) -> None:
    """Every model must return logits of shape [B, T, V]."""
    config = make_config(model_type)
    model = build_model(config).eval()
    x = torch.randint(0, VOCAB, (BATCH, SEQ))
    with torch.no_grad():
        out = model(x)
        if isinstance(model, DDMModel):
            logits, memory = out
            assert len(memory) == config.n_layers
            assert all(m.shape == (BATCH, config.d_model) for m in memory)
        else:
            logits = out
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_type", ["ddm", "ddm_ablation"])
def test_forward_shape_with_memory(model_type: str) -> None:
    """DDM forward must accept per-layer memory and keep output shape."""
    config = make_config(model_type)
    model = build_model(config).eval()
    x = torch.randint(0, VOCAB, (BATCH, SEQ))
    memory = [torch.zeros(BATCH, config.d_model) for _ in range(config.n_layers)]
    with torch.no_grad():
        logits, new_memory = model(x, memory=memory)
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert len(new_memory) == config.n_layers
    assert all(m.shape == (BATCH, config.d_model) for m in new_memory)
