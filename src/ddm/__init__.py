"""ddm: Distance-Decomposed Model -- a distance-aware language model.

Public API: model classes, the config dataclass and the training helpers.
"""

from __future__ import annotations

from ddm.config import DDMConfig
from ddm.models import (
    BigramModel,
    DDMModel,
    DistanceAwareAttention,
    DistanceAwareBlock,
    NGramModel,
    SmallTransformerLM,
    build_model,
    count_parameters,
)

__version__ = "0.1.0"

__all__ = [
    "BigramModel",
    "DDMConfig",
    "DDMModel",
    "DistanceAwareAttention",
    "DistanceAwareBlock",
    "NGramModel",
    "SmallTransformerLM",
    "__version__",
    "build_model",
    "count_parameters",
]
