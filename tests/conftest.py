"""Shared fixtures for the DDM test suite."""

from __future__ import annotations

import pytest
import torch

from ddm.config import DDMConfig

VOCAB: int = 1000
BATCH: int = 2
SEQ: int = 12


def make_config(model_type: str, **overrides: object) -> DDMConfig:
    """Build a small, fast DDMConfig for unit tests.

    Args:
        model_type: One of the supported model types.
        overrides: Extra config fields.

    Returns:
        A small test configuration.
    """
    defaults = {
        "vocab_size": VOCAB,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 64,
        "learn_g": model_type == "ddm",
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return DDMConfig(model_type=model_type, **defaults)


@pytest.fixture
def config() -> DDMConfig:
    """Default small DDM config."""
    return make_config("ddm")


@pytest.fixture
def x() -> torch.Tensor:
    """Random token ids."""
    return torch.randint(0, VOCAB, (BATCH, SEQ))


@pytest.fixture
def y() -> torch.Tensor:
    """Random target ids."""
    return torch.randint(0, VOCAB, (BATCH, SEQ))
