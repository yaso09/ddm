"""Config serialization: YAML round-trip must be lossless."""

from __future__ import annotations

from pathlib import Path

import pytest

from ddm.config import DDMConfig
from tests.conftest import make_config


def test_yaml_round_trip(tmp_path: Path) -> None:
    """to_yaml -> from_yaml must reproduce the identical config."""
    config = make_config("ddm", seeds=[0, 1], eval_buckets=[[0, 10], [50, 200]])
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = DDMConfig.from_yaml(path)
    assert loaded == config


def test_yaml_round_trip_all_model_types(tmp_path: Path) -> None:
    """Round-trip works for every model type."""
    for model_type in ("ddm", "ddm_ablation", "bigram", "ngram", "transformer"):
        config = make_config(model_type)
        path = tmp_path / f"{model_type}.yaml"
        config.to_yaml(path)
        assert DDMConfig.from_yaml(path) == config


def test_from_yaml_defaults_are_applied(tmp_path: Path) -> None:
    """Omitted YAML keys fall back to the dataclass defaults."""
    path = tmp_path / "minimal.yaml"
    path.write_text("model_type: bigram\n", encoding="utf-8")
    config = DDMConfig.from_yaml(path)
    assert config.model_type == "bigram"
    assert config.d_model == 128
    assert config.epochs == 5


def test_from_yaml_rejects_unknown_model_type(tmp_path: Path) -> None:
    """An invalid model_type raises ValueError."""
    path = tmp_path / "bad.yaml"
    path.write_text("model_type: unknown\n", encoding="utf-8")
    with pytest.raises(ValueError):
        DDMConfig.from_yaml(path)


def test_validation_d_model_divisible() -> None:
    """d_model % n_heads != 0 must raise ValueError."""
    with pytest.raises(ValueError):
        DDMConfig(model_type="ddm", d_model=33, n_heads=4)


def test_eval_buckets_normalized_to_tuples() -> None:
    """List buckets from YAML are normalized to tuples."""
    config = DDMConfig(model_type="ddm", eval_buckets=[[0, 10], [10, 50]])
    assert config.eval_buckets == [(0, 10), (10, 50)]


def test_repo_configs_parse(tmp_path: Path) -> None:
    """Every shipped config file parses cleanly."""
    repo = Path(__file__).resolve().parents[1]
    for config_file in sorted((repo / "configs").glob("*.yaml")):
        config = DDMConfig.from_yaml(config_file)
        assert config.model_type in ("ddm", "ddm_ablation", "bigram", "ngram", "transformer")