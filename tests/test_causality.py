"""Causality: outputs at position t must not depend on future tokens."""

from __future__ import annotations

import pytest
import torch

from ddm.models import DDMModel, build_model
from tests.conftest import VOCAB, make_config

ALL_TYPES = ["ddm", "ddm_ablation", "bigram", "ngram", "transformer"]

CUT = 6  # future positions start at CUT


def _logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    logits, _ = out if isinstance(model, DDMModel) else (out, None)
    return logits


@pytest.mark.parametrize("model_type", ALL_TYPES)
def test_future_inputs_do_not_affect_past_logits(model_type: str) -> None:
    """Corrupting x[:, CUT:] must leave logits[:, :CUT] untouched."""
    torch.manual_seed(0)
    config = make_config(model_type)
    model = build_model(config).eval()
    x = torch.randint(0, VOCAB, (1, 16))
    logits = _logits(model, x)

    x_corrupt = x.clone()
    x_corrupt[:, CUT:] = torch.randint(0, VOCAB, (1, 16 - CUT))
    logits_corrupt = _logits(model, x_corrupt)

    assert torch.allclose(
        logits[:, :CUT], logits_corrupt[:, :CUT], atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("model_type", ["ddm", "ddm_ablation"])
def test_future_inputs_do_not_affect_past_logits_with_memory(model_type: str) -> None:
    """The same guarantee must hold when a segment memory token is present."""
    torch.manual_seed(1)
    config = make_config(model_type)
    model = build_model(config).eval()
    x = torch.randint(0, VOCAB, (1, 16))
    memory = [
        torch.randn(1, config.d_model) for _ in range(config.n_layers)
    ]
    logits, _ = model(x, memory=memory)

    x_corrupt = x.clone()
    x_corrupt[:, CUT:] = torch.randint(0, VOCAB, (1, 16 - CUT))
    logits_corrupt, _ = model(x_corrupt, memory=memory)

    assert torch.allclose(
        logits[:, :CUT], logits_corrupt[:, :CUT], atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("model_type", ALL_TYPES)
def test_causal_strictness(model_type: str) -> None:
    """Position t's logits must actually change when x[:, t] changes."""
    torch.manual_seed(2)
    config = make_config(model_type)
    model = build_model(config).eval()
    x = torch.randint(0, VOCAB, (1, 16))
    logits = _logits(model, x)

    x_corrupt = x.clone()
    x_corrupt[:, CUT] = (x_corrupt[:, CUT] + 1) % VOCAB
    logits_corrupt = _logits(model, x_corrupt)

    assert not torch.allclose(
        logits[:, CUT], logits_corrupt[:, CUT], atol=1e-5, rtol=1e-5
    )
