"""Gradient flow: backward passes cleanly for every model."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ddm.models import DDMModel, build_model
from tests.conftest import VOCAB, make_config

ALL_TYPES = ["ddm", "ddm_ablation", "bigram", "ngram", "transformer"]


@pytest.mark.parametrize("model_type", ALL_TYPES)
def test_backward_all_parameters(model_type: str) -> None:
    """After loss.backward() every parameter has a finite gradient."""
    config = make_config(model_type)
    model = build_model(config)
    x = torch.randint(0, VOCAB, (BATCH := 2, SEQ := 16))
    y = torch.randint(0, VOCAB, (BATCH, SEQ))
    out = model(x)
    logits, _ = out if isinstance(model, DDMModel) else (out, None)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"NaN/inf gradient for {name}"
