"""Model zoo for the DDM project: model factory and exports."""

from __future__ import annotations

from torch import nn

from ddm.config import DDMConfig
from ddm.models.attention import DistanceAwareAttention
from ddm.models.baselines import BigramModel, NGramModel, SmallTransformerLM
from ddm.models.blocks import DistanceAwareBlock
from ddm.models.ddm_model import DDMModel


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters of a model.

    Args:
        model: Any PyTorch module.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: DDMConfig) -> nn.Module:
    """Instantiate the model described by a :class:`DDMConfig`.

    Args:
        config: Configuration whose ``model_type`` selects the model.

    Returns:
        The corresponding model: ``ddm``/``ddm_ablation`` build a
        :class:`DDMModel`, ``bigram`` a :class:`BigramModel`, ``ngram`` an
        :class:`NGramModel` and ``transformer`` a :class:`SmallTransformerLM`.
    """
    model_type = config.model_type
    if model_type in ("ddm", "ddm_ablation"):
        return DDMModel.from_config(config)
    if model_type == "bigram":
        return BigramModel(config.vocab_size, config.d_model, config.dropout)
    if model_type == "ngram":
        return NGramModel(
            config.vocab_size, config.d_model, config.n_context, config.dropout
        )
    if model_type == "transformer":
        return SmallTransformerLM(
            config.vocab_size,
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.max_seq_len,
            config.dropout,
        )
    raise ValueError(f"Unknown model_type {model_type!r}")


__all__ = [
    "BigramModel",
    "DDMModel",
    "DistanceAwareAttention",
    "DistanceAwareBlock",
    "NGramModel",
    "SmallTransformerLM",
    "build_model",
    "count_parameters",
]
