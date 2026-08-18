"""Autoregressive generation and checkpoint loading for trained models.

Shared by the ``ddm-chat`` REPL (``ddm.chat``) and ``07_Chat.ipynb``:
both call :func:`generate` / :func:`load_checkpoint`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ddm.config import DDMConfig
from ddm.data import _get_tokenizer
from ddm.models import (
    BigramModel,
    DDMModel,
    NGramModel,
    SmallTransformerLM,
    build_model,
)
from ddm.train import forward_step


@dataclass
class GenerationResult:
    """Output of :func:`generate`."""

    text: str
    token_ids: list[int]
    memory: list[torch.Tensor] | None
    tokens_per_sec: float


def _resolve_device(device: str) -> str:
    """Pick the compute device: explicit, else CUDA if available, else CPU."""
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _window_size(model: torch.nn.Module) -> int:
    """Context window the model actually conditions on per step."""
    if isinstance(model, (DDMModel, SmallTransformerLM)):
        return model.max_seq_len
    if isinstance(model, NGramModel):
        return model.n_context
    if isinstance(model, BigramModel):
        return 1
    return getattr(model, "max_seq_len", 128)


def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the ``top_k`` largest logits (others set to -inf)."""
    k = min(top_k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return torch.where(
        logits < kth, torch.full_like(logits, float("-inf")), logits
    )


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest set with cumulative prob >= top_p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    keep = torch.zeros_like(sorted_logits, dtype=torch.bool)
    keep[..., 0] = True
    keep[..., 1:] = cum_probs[..., :-1] <= top_p
    filtered = torch.where(keep, sorted_logits, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, sorted_indices, filtered)
    return out


def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    seed: int | None = None,
    device: str = "",
    memory: list[torch.Tensor] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> GenerationResult:
    """Sample token by token from a trained model.

    Args:
        model: Any supported model (ddm / ddm_ablation / transformer /
            bigram / ngram).
        tokenizer: Tokenizer whose ``__call__`` returns
            ``{"input_ids": [B, T]}``, with ``decode`` and ``eos_token_id``.
        prompt: Non-empty text to continue.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Softmax temperature (0 = greedy argmax).
        top_k: Keep only the k most likely tokens (0 = off).
        top_p: Nucleus mass (1.0 = off).
        seed: If set, seed torch's RNG for a reproducible draw.
        device: Compute device; empty string auto-detects (CUDA first).
        memory: DDM segment memory from the previous turn.
        on_token: Optional callback invoked with each decoded token.

    Returns:
        The generated result, including the new DDM memory to carry into
        the next turn.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    device = _resolve_device(device)
    window = _window_size(model)
    if memory is not None:
        memory = [m.detach().to(device) for m in memory]
    context: list[int] = tokenizer(prompt, return_tensors="pt")["input_ids"][
        0
    ].tolist()
    generated: list[int] = []
    t0 = time.time()
    with torch.no_grad():
        while len(generated) < max_new_tokens:
            x = torch.tensor([context[-window:]], dtype=torch.long, device=device)
            logits, memory = forward_step(model, x, memory)
            scores = logits[0, -1]
            if temperature > 0:
                scores = scores / temperature
            if top_k > 0:
                scores = _top_k_filter(scores, top_k)
            if top_p < 1.0:
                scores = _top_p_filter(scores, top_p)
            if temperature > 0:
                probs = torch.softmax(scores, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            else:
                next_id = int(scores.argmax().item())
            if next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)
            context.append(next_id)
            if on_token is not None:
                on_token(tokenizer.decode([next_id]))
    elapsed = max(time.time() - t0, 1e-9)
    return GenerationResult(
        text=tokenizer.decode(generated),
        token_ids=generated,
        memory=memory,
        tokens_per_sec=len(generated) / elapsed,
    )


def load_checkpoint(
    path: str | Path, device: str = ""
) -> tuple[torch.nn.Module, DDMConfig, Any]:
    """Load a checkpoint written by ``run_training`` (see ``ddm.train``).

    Args:
        path: Path to the ``{model}_seed{n}.pt`` file.
        device: Compute device; empty string auto-detects.

    Returns:
        Tuple ``(model, config, tokenizer)`` with the model in ``eval()``
        mode on the target device.
    """
    device = _resolve_device(device)
    raw = torch.load(path, map_location=device, weights_only=True)
    config = DDMConfig(**raw["config"])
    model = build_model(config)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    return model, config, _get_tokenizer()