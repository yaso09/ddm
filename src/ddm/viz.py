"""Plotting helpers for DDM experiments (saved to disk, no display needed)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_g_curve(
    k: np.ndarray,
    curves: np.ndarray,
    save_path: str | Path,
    title: str = "Distance gate g(k)",
) -> str:
    """Plot the learned distance gate g(k) per layer and save it.

    Args:
        k: Distances ``[max_k]`` (x axis).
        curves: Gate values ``[n_layers, max_k]`` (one row per layer).
        save_path: Destination PNG path.
        title: Plot title.

    Returns:
        The path the plot was written to.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for layer, values in enumerate(curves):
        ax.plot(k, values, marker="o", ms=3, label=f"layer {layer}")
    ax.set_xlabel("distance k")
    ax.set_ylabel("g(k)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(k.min(), k.max())
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_loss_curve(
    train_losses: Sequence[float],
    val_ppls: Sequence[float],
    save_path: str | Path,
    title: str = "Training curves",
) -> str:
    """Plot training loss and validation perplexity per epoch.

    Args:
        train_losses: Mean training loss per epoch.
        val_ppls: Validation perplexity per epoch.
        save_path: Destination PNG path.
        title: Plot title.

    Returns:
        The path the plot was written to.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(epochs, train_losses, marker="o", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_ppls, marker="s", color="tab:orange", label="val PPL")
    ax2.set_ylabel("val PPL", color="tab:orange")
    fig.suptitle(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_bucket_ppl(
    labels: list[str],
    values: list[float],
    save_path: str | Path,
    title: str = "Perplexity by position bucket",
) -> str:
    """Bar plot of perplexity across position buckets.

    Args:
        labels: Bucket labels (e.g. ``"0-10"``).
        values: Perplexity per bucket.
        save_path: Destination PNG path.
        title: Plot title.

    Returns:
        The path the plot was written to.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values)
    ax.set_xlabel("position bucket")
    ax.set_ylabel("perplexity")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)
