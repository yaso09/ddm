"""Training, evaluation and result-collection for DDM experiments."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ddm import viz
from ddm.config import DDMConfig
from ddm.data import load_and_tokenize, make_loader
from ddm.models import DDMModel, build_model, count_parameters


def set_seed(seed: int) -> None:
    """Seed all random generators for reproducibility.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_step(
    model: nn.Module,
    x: torch.Tensor,
    memory: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Run a model forward pass, handling the DDM return signature.

    Args:
        model: Any supported model.
        x: Input token ids ``[B, T]``.
        memory: Optional DDM segment memory (ignored by baselines).

    Returns:
        Tuple ``(logits, new_memory)``; ``new_memory`` is ``None`` for
        baselines.
    """
    if isinstance(model, DDMModel):
        return model(x, memory=memory)
    return model(x), None


def _nll_sum(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Summed per-token negative log-likelihood.

    Args:
        logits: Logits ``[B, T, V]``.
        y: Target ids ``[B, T]``.

    Returns:
        Scalar NLL sum.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    memory: list[torch.Tensor] | None = None,
    max_steps: int | None = None,
    log_every: int = 50,
) -> tuple[float, int, list[torch.Tensor] | None]:
    """Train for one epoch, carrying segment memory across batches.

    Args:
        model: The model to train.
        loader: Training DataLoader (shuffle=False so blocks follow document
            order and DDM segment memory stays meaningful).
        optimizer: Optimizer.
        device: Compute device.
        memory: Initial DDM segment memory (None at epoch start).
        max_steps: Optional cap on steps per epoch.
        log_every: Print mean loss every N steps.

    Returns:
        Tuple ``(mean_loss, steps, new_memory)``.
    """
    model.train()
    total_loss = 0.0
    steps = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, memory = forward_step(model, x, memory)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
        if steps % log_every == 0:
            print(f"    step {steps}: loss {total_loss / steps:.4f}")
        if max_steps is not None and steps >= max_steps:
            break
    return total_loss / steps, steps, memory


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    memory: list[torch.Tensor] | None = None,
) -> tuple[float, list[torch.Tensor] | None]:
    """Compute mean perplexity over a split.

    Args:
        model: The model.
        loader: Evaluation DataLoader.
        device: Compute device.
        memory: Optional initial DDM segment memory.

    Returns:
        Tuple ``(perplexity, final_memory)``.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, memory = forward_step(model, x, memory)
        total_nll += _nll_sum(logits, y).item()
        total_tokens += y.numel()
    ppl = math.exp(total_nll / total_tokens)
    return ppl, memory


@torch.no_grad()
def evaluate_by_position_bucket(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    buckets: list[tuple[int, int]],
    memory: list[torch.Tensor] | None = None,
) -> tuple[dict[tuple[int, int], float], list[torch.Tensor] | None]:
    """Compute perplexity per position bucket.

    Buckets slice the *target* positions (e.g. ``(0, 10)`` covers the first
    ten tokens). Comparing buckets shows whether a model actually benefits
    from longer context: a lower PPL on late positions means the long
    history helps.

    Args:
        model: The model.
        loader: Evaluation DataLoader.
        device: Compute device.
        buckets: List of ``(start, end)`` position ranges.
        memory: Optional initial DDM segment memory.

    Returns:
        Tuple ``(ppl_by_bucket, final_memory)`` where ``ppl_by_bucket`` maps
        each bucket (as a tuple) to its perplexity.
    """
    model.eval()
    nll_sums: dict[tuple[int, int], float] = {b: 0.0 for b in buckets}
    nll_counts: dict[tuple[int, int], int] = {b: 0 for b in buckets}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, memory = forward_step(model, x, memory)
        per_token = F.cross_entropy(
            logits.transpose(1, 2), y, reduction="none"
        )
        T = y.size(1)
        for bucket in buckets:
            lo, hi = bucket
            hi = min(hi, T)
            if lo < hi:
                part = per_token[:, lo:hi]
                nll_sums[bucket] += part.sum().item()
                nll_counts[bucket] += part.numel()
    ppl: dict[tuple[int, int], float] = {
        b: math.exp(nll_sums[b] / nll_counts[b])
        for b in buckets
        if nll_counts[b] > 0
    }
    return ppl, memory


def _pick_device(config: DDMConfig) -> torch.device:
    """Choose the compute device.

    Args:
        config: Configuration (``device`` may be empty for auto-detect).

    Returns:
        The selected device.
    """
    if config.device:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(config: DDMConfig) -> dict[str, Any]:
    """Run the full training loop described by a config.

    For every seed in ``config.seeds``: seed everything, build the model,
    train for ``config.epochs``, evaluate on validation each epoch and on
    test at the end (mean PPL + position-bucket PPL). Artifacts are written
    into ``config.save_dir``: the checkpoint ``{model}_seed{n}.pt``, the
    learned g(k) curve (``.png``/``.npz``) for DDM models and a loss/val
    curve plot.

    Args:
        config: Full experiment configuration.

    Returns:
        A results dictionary with keys ``model_type``, ``n_params``,
        ``per_seed`` (list of per-seed metrics) and aggregated ``mean``/``std``
        entries (test PPL and bucket PPLs).
    """
    device = _pick_device(config)
    print(f"[train] device: {device}")

    blocks, vocab_size, _ = load_and_tokenize(
        seq_len=config.max_seq_len,
        cache_dir=config.cache_dir,
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        max_blocks=config.max_blocks,
    )
    config.vocab_size = vocab_size
    print(f"[train] vocab_size: {vocab_size}")

    train_loader = make_loader(blocks["train"], config.batch_size, config.shuffle)
    val_loader = make_loader(blocks["val"], config.batch_size, shuffle=False)
    test_loader = make_loader(blocks["test"], config.batch_size, shuffle=False)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, Any]] = []
    for seed in config.seeds:
        print(f"\n=== seed {seed} ===")
        set_seed(seed)
        model = build_model(config).to(device)
        n_params = count_parameters(model)
        print(f"[train] {config.model_type} parameters: {n_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        train_losses: list[float] = []
        val_ppls: list[float] = []
        t0 = time.time()
        for epoch in range(config.epochs):
            print(f"--- epoch {epoch + 1}/{config.epochs} ---")
            loss, _, _ = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                max_steps=config.max_steps,
                log_every=config.log_every,
            )
            train_losses.append(loss)
            val_ppl, _ = evaluate(model, val_loader, device)
            val_ppls.append(val_ppl)
            print(f"[train] epoch {epoch + 1}: loss {loss:.4f}, val PPL {val_ppl:.2f}")

        test_ppl, _ = evaluate(model, test_loader, device)
        bucket_ppl, _ = evaluate_by_position_bucket(
            model, test_loader, device, config.eval_buckets
        )
        wall_time = time.time() - t0
        print(f"[train] test PPL: {test_ppl:.2f} ({wall_time:.1f}s)")

        stem = f"{config.model_type}_seed{seed}"
        ckpt_path = save_dir / f"{stem}.pt"
        torch.save(
            {"state_dict": model.state_dict(), "config": asdict(config)},
            ckpt_path,
        )
        g_curve_path: str | None = None
        if isinstance(model, DDMModel):
            k, curves = model.get_g_curve(max_k=config.max_seq_len, device="")
            np.savez(save_dir / f"g_curve_{stem}.npz", k=k, g=curves)
            g_curve_path = viz.save_g_curve(
                k,
                curves,
                save_dir / f"g_curve_{stem}.png",
                title=f"Distance gate g(k) -- {config.model_type} (seed {seed})",
            )
        loss_curve_path = viz.plot_loss_curve(
            train_losses,
            val_ppls,
            save_dir / f"loss_curve_{stem}.png",
            title=f"Training curves -- {config.model_type} (seed {seed})",
        )

        per_seed.append(
            {
                "seed": seed,
                "train_loss": train_losses,
                "val_ppl": val_ppls,
                "test_ppl": test_ppl,
                "buckets_ppl": {f"{lo}-{hi}": v for (lo, hi), v in bucket_ppl.items()},
                "wall_time_s": wall_time,
                "train_steps": len(train_loader) * config.epochs
                if config.max_steps is None
                else config.max_steps * config.epochs,
                "checkpoint": str(ckpt_path),
                "g_curve": g_curve_path,
                "loss_curve": loss_curve_path,
            }
        )

    def _mean(key: str) -> float:
        return float(np.mean([p[key] for p in per_seed]))

    def _std(key: str) -> float:
        return float(np.std([p[key] for p in per_seed])) if len(per_seed) > 1 else 0.0

    bucket_keys = [f"{lo}-{hi}" for lo, hi in config.eval_buckets]
    results: dict[str, Any] = {
        "model_type": config.model_type,
        "n_params": n_params,
        "per_seed": per_seed,
        "test_ppl_mean": _mean("test_ppl"),
        "test_ppl_std": _std("test_ppl"),
        "wall_time_mean_s": _mean("wall_time_s"),
        "buckets_ppl_mean": {
            key: float(np.mean([p["buckets_ppl"][key] for p in per_seed]))
            for key in bucket_keys
        },
    }
    return results


def format_results_table(results: dict[str, Any]) -> str:
    """Render a results dictionary as a markdown table.

    Args:
        results: Output of :func:`run_training`.

    Returns:
        Markdown table string.
    """
    header = (
        "| Seed | Test PPL | "
        + " | ".join(f"PPL({key})" for key in results["buckets_ppl_mean"])
        + " | Time (s) |"
    )
    sep = "|" + "---|" * (4 + len(results["buckets_ppl_mean"]))
    rows = [header, sep]
    for entry in results["per_seed"]:
        row = (
            f"| {entry['seed']} | {entry['test_ppl']:.2f} | "
            + " | ".join(f"{entry['buckets_ppl'][key]:.2f}" for key in results["buckets_ppl_mean"])
            + f" | {entry['wall_time_s']:.1f} |"
        )
        rows.append(row)
    mean_row = (
        f"| mean | {results['test_ppl_mean']:.2f} ± {results['test_ppl_std']:.2f} | "
        + " | ".join(f"{v:.2f}" for v in results["buckets_ppl_mean"].values())
        + f" | {results['wall_time_mean_s']:.1f} |"
    )
    rows.append(mean_row)
    return "\n".join(rows) + "\n"
