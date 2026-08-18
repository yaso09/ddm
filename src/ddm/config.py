"""Configuration for DDM models and training runs.

Configs are serialized to/from YAML so that every experiment can be
captured in a single, reproducible file (see ``configs/``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

MODEL_TYPES: tuple[str, ...] = (
    "ddm",
    "ddm_ablation",
    "bigram",
    "ngram",
    "transformer",
)


@dataclass
class DDMConfig:
    """Complete description of a model architecture and its training run.

    Attributes:
        model_type: One of ``MODEL_TYPES``. ``ddm`` is the proposed model
            (learned distance gate), ``ddm_ablation`` freezes the gate to
            ``1/k``, the rest are baselines.
        vocab_size: Vocabulary size (from the tokenizer).
        d_model: Hidden/embedding dimension.
        n_layers: Number of blocks (DDM/transformer) or unused otherwise.
        n_heads: Number of attention heads (DDM/transformer).
        max_seq_len: Sequence length; also normalizes distances for g(k).
        n_context: Context size for the n-gram model (``n``).
        learn_g: Whether the distance gate g(k) is learned. When False the
            gate is fixed to ``1/k`` (ablation).
        dropout: Dropout probability used in model blocks.
        batch_size: Training batch size.
        lr: AdamW learning rate.
        epochs: Number of training epochs.
        seeds: Random seeds; each seed yields one full training run.
        dataset_name: HuggingFace dataset name (e.g. ``wikitext``).
        dataset_config: Dataset configuration (e.g. ``wikitext-2-raw-v1``).
        cache_dir: Directory where the dataset is cached.
        shuffle: Whether training batches are shuffled. Keep False so that
            blocks follow document order and segment memory is meaningful.
        eval_buckets: List of ``(start, end)`` position buckets used by
            ``evaluate_by_position_bucket``.
        log_every: Print training loss every N steps.
        save_dir: Directory for checkpoints, curves and result tables.
        max_steps: Cap on training steps per epoch (None = unlimited).
        max_blocks: Cap on the number of blocks kept per split (None =
            unlimited); useful for smoke runs.
        device: Torch device string; empty means auto-detect.
    """

    model_type: str = "ddm"
    vocab_size: int = 50257
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    max_seq_len: int = 128
    n_context: int = 3
    learn_g: bool = True
    dropout: float = 0.0
    batch_size: int = 32
    lr: float = 3e-4
    epochs: int = 5
    seeds: list[int] = field(default_factory=lambda: [0])
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    cache_dir: str = "hf_cache"
    shuffle: bool = False
    eval_buckets: list[tuple[int, int]] = field(
        default_factory=lambda: [(0, 10), (10, 50), (50, 200)]
    )
    log_every: int = 50
    save_dir: str = "checkpoints"
    max_steps: int | None = None
    max_blocks: int | None = None
    device: str = ""

    def __post_init__(self) -> None:
        """Validate the configuration and normalize nested types."""
        if self.model_type not in MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type {self.model_type!r}; expected one of {MODEL_TYPES}"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.eval_buckets = [tuple(b) for b in self.eval_buckets]

    @classmethod
    def from_yaml(cls, path: str | Path) -> DDMConfig:
        """Load a configuration from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            The parsed configuration.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize the configuration to a YAML file.

        Args:
            path: Destination path for the YAML file.
        """
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False)
