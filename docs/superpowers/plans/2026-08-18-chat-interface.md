# Chat Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user converse with any trained checkpoint from the terminal (`ddm-chat`) and from a notebook (`07_Chat.ipynb`), sharing one generation core.

**Architecture:** A new `src/ddm/generate.py` provides `load_checkpoint()` and a sampling `generate()` (temperature / top-k / top-p, EOS stop, DDM segment-memory carry, streaming callback, GPU auto-detect). `src/ddm/chat.py` wraps it in a REPL with `/help`, `/reset`, `/quit`. The notebook calls the same functions. Tests force CPU.

**Tech Stack:** Python 3.10+, PyTorch, GPT-2 BPE tokenizer (transformers), argparse, pytest, ruff, jupyter/nbconvert.

**Spec:** `docs/superpowers/specs/2026-08-18-chat-interface-design.md`

---

### Task 1: Generation core (`src/ddm/generate.py`)

**Files:**
- Create: `src/ddm/generate.py`
- Test: `tests/test_generate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate.py`:

```python
"""Generation core: sampling filters, EOS handling, DDM memory carry, checkpoint loading."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

import ddm.generate as gen
from ddm.models import DDMModel, build_model
from tests.conftest import make_config


class FakeTokenizer:
    """Offline stand-in for the GPT-2 tokenizer (27-token vocab, EOS = 26)."""

    eos_token_id = 26

    def __call__(self, text, return_tensors=None):
        ids = [min(ord(c) % 26, 25) for c in text] or [0]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids):
        return "".join(chr(65 + i) for i in ids)


class FakeModel(torch.nn.Module):
    """Model with fixed logits (input ignored); used to exercise the filters."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("w", logits)

    def forward(self, x):
        return self.w.unsqueeze(0).expand(x.shape[0], x.shape[1], -1)


def _fixed_logits() -> torch.Tensor:
    """Random logits with the EOS index (26) suppressed to -1e9."""
    torch.manual_seed(0)
    logits = torch.randn(27)
    logits[26] = -1e9
    return logits


def test_same_seed_is_deterministic() -> None:
    model = FakeModel(_fixed_logits())
    tok = FakeTokenizer()
    a = gen.generate(model, tok, "ab", max_new_tokens=10, seed=42, device="cpu")
    b = gen.generate(model, tok, "ab", max_new_tokens=10, seed=42, device="cpu")
    assert a.token_ids == b.token_ids
    c = gen.generate(model, tok, "ab", max_new_tokens=10, seed=43, device="cpu")
    assert a.token_ids != c.token_ids


def test_max_new_tokens_cap() -> None:
    logits = torch.arange(27, dtype=torch.float32)
    logits[26] = -1e9
    model = FakeModel(logits)
    tok = FakeTokenizer()
    out = gen.generate(model, tok, "ab", max_new_tokens=7, temperature=0, device="cpu")
    assert out.token_ids == [25] * 7
    assert out.text == "Z" * 7


def test_eos_stops_early() -> None:
    logits = torch.zeros(27)
    logits[26] = 10.0
    model = FakeModel(logits)
    tok = FakeTokenizer()
    out = gen.generate(model, tok, "ab", max_new_tokens=20, temperature=0, device="cpu")
    assert out.token_ids == []
    assert out.text == ""


def test_temperature_zero_is_greedy() -> None:
    logits = torch.tensor([0.1, 0.5, 0.3] + [0.0] * 23 + [-1e9])
    model = FakeModel(logits)
    tok = FakeTokenizer()
    out = gen.generate(model, tok, "ab", max_new_tokens=5, temperature=0, device="cpu")
    assert out.token_ids == [1] * 5


def test_top_k_one_keeps_only_argmax() -> None:
    logits = torch.tensor([0.3, 0.5, 0.1] + [0.0] * 23 + [-1e9])
    model = FakeModel(logits)
    tok = FakeTokenizer()
    out = gen.generate(
        model, tok, "ab", max_new_tokens=4, top_k=1, top_p=1.0, device="cpu"
    )
    assert out.token_ids == [1] * 4


def test_top_p_very_small_keeps_only_top_token() -> None:
    logits = torch.tensor([0.3, 0.2, 0.1] + [0.0] * 23 + [-1e9])
    model = FakeModel(logits)
    tok = FakeTokenizer()
    out = gen.generate(
        model, tok, "ab", max_new_tokens=3, top_k=0, top_p=1e-9, device="cpu"
    )
    assert out.token_ids == [0] * 3


def test_ddm_memory_carried_between_turns() -> None:
    torch.manual_seed(11)
    config = make_config("ddm", vocab_size=27, n_layers=2, max_seq_len=32)
    model = DDMModel.from_config(config)
    tok = FakeTokenizer()
    first = gen.generate(
        model, tok, "ab", max_new_tokens=8, temperature=0, seed=3, device="cpu"
    )
    assert first.memory is not None
    assert [tuple(m.shape) for m in first.memory] == [
        (1, config.d_model),
        (1, config.d_model),
    ]
    continued = gen.generate(
        model,
        tok,
        "ab",
        max_new_tokens=8,
        temperature=0,
        seed=3,
        device="cpu",
        memory=first.memory,
    )
    fresh = gen.generate(
        model, tok, "ab", max_new_tokens=8, temperature=0, seed=3, device="cpu"
    )
    assert continued.text != fresh.text


def test_baseline_returns_no_memory() -> None:
    config = make_config("ngram", vocab_size=27, n_context=3)
    model = build_model(config)
    tok = FakeTokenizer()
    out = gen.generate(model, tok, "abc", max_new_tokens=5, temperature=0, device="cpu")
    assert out.memory is None
    assert len(out.token_ids) == 5


def test_load_checkpoint_roundtrip(tmp_path, monkeypatch) -> None:
    config = make_config("bigram", vocab_size=27)
    model = build_model(config)
    path = tmp_path / "bigram_seed0.pt"
    torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, path)
    monkeypatch.setattr(gen, "_get_tokenizer", lambda: FakeTokenizer())
    loaded, loaded_config, tokenizer = gen.load_checkpoint(str(path), device="cpu")
    assert loaded_config.model_type == "bigram"
    assert set(loaded.state_dict()) == set(model.state_dict())
    assert tokenizer.eos_token_id == 26


def test_empty_prompt_rejected() -> None:
    model = FakeModel(_fixed_logits())
    tok = FakeTokenizer()
    with pytest.raises(ValueError):
        gen.generate(model, tok, "   ", device="cpu")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ddm.generate'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/ddm/generate.py`:

```python
"""Autoregressive generation and checkpoint loading for trained models.

Shared by the ``ddm-chat`` REPL (``ddm.chat``) and ``07_Chat.ipynb``:
both call :func:`generate` / :func:`load_checkpoint`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from ddm.config import DDMConfig
from ddm.data import _get_tokenizer
from ddm.models import (
    DDMModel,
    NGramModel,
    SmallTransformerLM,
    BigramModel,
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
    raw = torch.load(path, map_location=device)
    config = DDMConfig(**raw["config"])
    model = build_model(config)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    return model, config, _get_tokenizer()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_generate.py -v`
Expected: 10 passed

- [ ] **Step 5: Lint and commit**

Run: `.venv\Scripts\python.exe -m ruff check src/ddm/generate.py tests/test_generate.py`
Expected: no errors

```bash
git add src/ddm/generate.py tests/test_generate.py
git commit -m "feat: add autoregressive generation core with sampling filters"
```

---

### Task 2: Interactive REPL (`src/ddm/chat.py`)

**Files:**
- Create: `src/ddm/chat.py`
- Modify: `pyproject.toml:42-43` (add the `ddm-chat` script entry)
- Test: `tests/test_chat.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat.py`:

```python
"""ddm-chat REPL: smoke test and error handling."""

from __future__ import annotations

import io
import sys
from dataclasses import asdict

import torch

import ddm.generate as gen
from ddm.chat import main
from ddm.models import build_model
from tests.conftest import make_config
from tests.test_generate import FakeTokenizer


def _write_checkpoint(path, config) -> None:
    model = build_model(config)
    torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, path)


def test_main_smoke(tmp_path, monkeypatch, capsys) -> None:
    config = make_config("bigram", vocab_size=27)
    ckpt = tmp_path / "bigram_seed0.pt"
    _write_checkpoint(ckpt, config)
    monkeypatch.setattr(gen, "_get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello world\n/reset\n/quit\n"))
    rc = main(
        [
            "--checkpoint",
            str(ckpt),
            "--max-new-tokens",
            "5",
            "--seed",
            "1",
            "--device",
            "cpu",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "ddm-chat: bigram" in out
    assert "[reset] context cleared" in out
    assert "tokens" in out


def test_unknown_checkpoint_exits_2(tmp_path, capsys) -> None:
    rc = main(["--checkpoint", str(tmp_path / "missing.pt"), "--device", "cpu"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "cannot load" in err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ddm.chat'`

- [ ] **Step 3: Write the implementation**

Create `src/ddm/chat.py`:

```python
"""Interactive chat REPL over trained checkpoints: ``ddm-chat``."""

from __future__ import annotations

import argparse
import sys

import torch

from ddm.generate import generate, load_checkpoint

HELP_TEXT = """commands:
  /help   show this help
  /reset  clear conversation context and DDM memory
  /quit   exit the session (Ctrl-D / Ctrl-C also work)"""


def build_parser() -> argparse.ArgumentParser:
    """Create the ``ddm-chat`` argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="ddm-chat",
        description="Chat with a trained DDM project checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ddm_seed0.pt",
        help="Path to a checkpoint (default: checkpoints/ddm_seed0.pt).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Tokens per turn."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 = greedy).",
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k sampling (0 = off)."
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling (1 = off)."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducible turns."
    )
    parser.add_argument(
        "--device", default="", help="Torch device (default: auto)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ddm-chat`` console script.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 on success, 2 on checkpoint errors).
    """
    args = build_parser().parse_args(argv)
    try:
        model, config, tokenizer = load_checkpoint(args.checkpoint, device=args.device)
    except Exception as exc:  # noqa: BLE001 - surface any load failure
        print(
            f"[ddm-chat] cannot load checkpoint {args.checkpoint}: {exc}",
            file=sys.stderr,
        )
        return 2
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"ddm-chat: {config.model_type} | d_model {config.d_model} | "
        f"device {device} | type /help"
    )
    history = ""
    memory = None
    try:
        while True:
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                print()
                break
            if line == "":
                break
            line = line.strip()
            if not line:
                continue
            if line == "/quit":
                break
            if line == "/help":
                print(HELP_TEXT)
                continue
            if line == "/reset":
                history = ""
                memory = None
                print("[ddm-chat] context cleared")
                continue
            if line.startswith("/"):
                print(f"[ddm-chat] unknown command {line!r} (type /help)")
                continue
            history += line + "\n"
            result = generate(
                model,
                tokenizer,
                history,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
                device=args.device,
                memory=memory,
                on_token=lambda tok: print(tok, end="", flush=True),
            )
            print()
            memory = result.memory
            history += result.text + "\n"
            note = " | memory carried" if result.memory is not None else ""
            print(
                f"[{len(result.token_ids)} tokens, "
                f"{result.tokens_per_sec:.1f} tok/s{note}]"
            )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Edit `pyproject.toml`, after the `ddm-train` line:

```toml
[project.scripts]
ddm-train = "ddm.cli:main"
ddm-chat = "ddm.chat:main"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chat.py -v`
Expected: 2 passed

- [ ] **Step 5: Manual smoke test with a real checkpoint**

Run (Windows PowerShell):

```powershell
"hello world", "/quit" | & .venv\Scripts\python.exe -m ddm.chat --checkpoint checkpoints/bigram_seed0.pt --max-new-tokens 20 --seed 1 --device cpu
```

Expected: banner line (`ddm-chat: bigram | ...`), a streamed continuation, a `[N tokens, ...]` summary, exit 0.

- [ ] **Step 6: Lint and commit**

Run: `.venv\Scripts\python.exe -m ruff check src/ddm/chat.py tests/test_chat.py`
Expected: no errors

```bash
git add src/ddm/chat.py tests/test_chat.py pyproject.toml
git commit -m "feat: add ddm-chat interactive REPL"
```

---

### Task 3: Notebook (`notebooks/07_Chat.ipynb`)

**Files:**
- Create: `notebooks/07_Chat.ipynb`

- [ ] **Step 1: Write the notebook**

Create `notebooks/07_Chat.ipynb` with the following exact JSON:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 07 — Chatting with the trained models\n",
    "\n",
    "This notebook exercises the same `generate()` / `load_checkpoint()` API that powers the `ddm-chat` terminal REPL (`src/ddm/generate.py`). The device is auto-detected: CUDA when available, CPU otherwise."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "\n",
    "from ddm.generate import generate, load_checkpoint\n",
    "\n",
    "CKPT_DIR = Path(\"checkpoints\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "checkpoints = sorted(CKPT_DIR.glob(\"*_seed*.pt\"))\n",
    "print(\"found:\", [p.name for p in checkpoints])\n",
    "\n",
    "models = {}\n",
    "for path in checkpoints:\n",
    "    try:\n",
    "        model, config, tok = load_checkpoint(str(path))\n",
    "    except Exception as exc:\n",
    "        print(f\"skip {path.name}: {exc}\")\n",
    "        continue\n",
    "    models[path.stem] = (model, config)\n",
    "\n",
    "if not models:\n",
    "    raise SystemExit(\"no checkpoints loaded\")\n",
    "\n",
    "print(\"loaded:\", sorted(models))\n",
    "device = next(next(iter(models.values()))[0].parameters()).device\n",
    "print(\"device:\", device)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. One-shot generation across models\n",
    "\n",
    "The same prompt is continued by every trained model. DDM and the transformer see the whole prompt; bigram conditions on a single token and 3-gram on three, which is why they drift quickly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"The theory of relativity was developed by\"\n",
    "for name, (model, config) in models.items():\n",
    "    out = generate(model, tok, prompt, max_new_tokens=50, seed=7)\n",
    "    print(f\"\\n--- {name} ---\\n{prompt}{out.text}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Sampling parameters\n",
    "\n",
    "`temperature` sharpens or flattens the distribution; `top_k` keeps only the k most likely tokens. Same seed, same prompt, same model — only the sampler changes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "name = \"ddm_seed0\"\n",
    "model, config = models[name]\n",
    "base = \"The castle stood on the hill\"\n",
    "\n",
    "for temp in (0.2, 0.8, 1.5):\n",
    "    out = generate(model, tok, base, max_new_tokens=40, temperature=temp,\n",
    "                   top_k=0, top_p=1.0, seed=7)\n",
    "    print(f\"\\n--- temperature {temp} ---\\n{base}{out.text}\")\n",
    "\n",
    "for top_k in (1, 10, 50):\n",
    "    out = generate(model, tok, base, max_new_tokens=40, temperature=0.8,\n",
    "                   top_k=top_k, top_p=1.0, seed=7)\n",
    "    print(f\"\\n--- top_k {top_k} ---\\n{base}{out.text}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Long context: DDM segment memory\n",
    "\n",
    "The story below is longer than the 128-token window both models were trained with. DDM keeps a per-layer segment memory of everything before the window, so content from the beginning of the story can still influence the continuation; the transformer must discard it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "story = (\n",
    "    \"The northern kingdom was founded by King Aldric in the year 812. \"\n",
    "    \"His daughter, Queen Elara, expanded the borders to the sea. \"\n",
    "    \"The royal library held three thousand volumes of history. \"\n",
    "    \"After the great fire of 941, the library was rebuilt in stone. \"\n",
    "    \"Aldric's great-grandson, King Bram, signed the treaty of the bay. \"\n",
    "    \"The treaty ended the war with the southern states. \"\n",
    "    \"Bram's advisor was a woman named Selma, who kept the royal seal. \"\n",
    "    \"She hid the seal in the hollow of an old oak tree. \"\n",
    "    \"In the winter of 977, the seal was stolen by a thief named Dorian.\"\n",
    ")\n",
    "follow_up = \" The royal seal was hidden\"\n",
    "\n",
    "story_ids = tok(story, return_tensors=\"pt\")[\"input_ids\"][0]\n",
    "print(f\"story length: {len(story_ids)} tokens \"\n",
    "      f\"(model window: {models['ddm_seed0'][1].max_seq_len})\")\n",
    "\n",
    "for name in (\"ddm_seed0\", \"transformer_seed0\"):\n",
    "    if name not in models:\n",
    "        print(f\"{name} not available, skipping\")\n",
    "        continue\n",
    "    model, cfg = models[name]\n",
    "    out = generate(model, tok, story + follow_up, max_new_tokens=40,\n",
    "                   temperature=0.8, seed=7)\n",
    "    print(f\"\\n--- {name} ---\\n{story + follow_up}{out.text}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Interactive chat widget\n",
    "\n",
    "A small ipywidgets panel. Each click runs `generate()` on the selected model with streaming output; the terminal `ddm-chat` REPL behaves the same way."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    import ipywidgets as widgets\n",
    "    from IPython.display import display\n",
    "except ImportError:\n",
    "    widgets = None\n",
    "\n",
    "if widgets is None:\n",
    "    print(\"ipywidgets not installed - run: uv pip install ipywidgets\")\n",
    "else:\n",
    "    model_menu = widgets.Dropdown(options=sorted(models), value=\"ddm_seed0\",\n",
    "                                  description=\"model\")\n",
    "    max_tok = widgets.IntSlider(value=80, min=8, max=256, step=8,\n",
    "                                description=\"max tokens\")\n",
    "    temp = widgets.FloatSlider(value=0.8, min=0.0, max=2.0, step=0.05,\n",
    "                               description=\"temperature\")\n",
    "    prompt_box = widgets.Textarea(placeholder=\"Type a prompt...\",\n",
    "                                  description=\"prompt\",\n",
    "                                  layout=widgets.Layout(width=\"100%\",\n",
    "                                                       height=\"90px\"))\n",
    "    run_btn = widgets.Button(description=\"Generate\")\n",
    "    out = widgets.Output(layout=widgets.Layout(width=\"100%\", height=\"300px\",\n",
    "                                               overflow_y=\"auto\"))\n",
    "\n",
    "    def on_run(_):\n",
    "        with out:\n",
    "            out.clear_output()\n",
    "            text = prompt_box.value.strip()\n",
    "            if not text:\n",
    "                print(\"(empty prompt)\")\n",
    "                return\n",
    "            model, cfg = models[model_menu.value]\n",
    "            print(f\"> {text}\\n\")\n",
    "            result = generate(\n",
    "                model, tok, text,\n",
    "                max_new_tokens=max_tok.value,\n",
    "                temperature=temp.value,\n",
    "                on_token=lambda t: print(t, end=\"\", flush=True),\n",
    "            )\n",
    "            print(f\"\\n[{len(result.token_ids)} tokens, \"\n",
    "                  f\"{result.tokens_per_sec:.1f} tok/s]\")\n",
    "\n",
    "    run_btn.on_click(on_run)\n",
    "    display(widgets.VBox([model_menu, max_tok, temp, prompt_box, run_btn, out]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Summary\n",
    "\n",
    "| Model | Context per step | Memory across turns |\n",
    "|---|---|---|\n",
    "| ddm | last 128 tokens | segment memory carried |\n",
    "| ddm_ablation | last 128 tokens | segment memory carried (fixed 1/k gate) |\n",
    "| transformer | last 128 tokens | none |\n",
    "| bigram | 1 token | none |\n",
    "| ngram | last 3 tokens | none |"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Execute the notebook in place**

Run: `.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/07_Chat.ipynb`
(If `jupyter` is missing: `.venv\Scripts\python.exe -m pip install "jupyter" "ipywidgets"` first.)
Expected: all cells execute without error; generated text appears in cell outputs.

- [ ] **Step 3: Verify the outputs are saved**

Run: `.venv\Scripts\python.exe -c "import json,pathlib; nb=json.loads(pathlib.Path('notebooks/07_Chat.ipynb').read_text(encoding='utf-8')); cells=[c for c in nb['cells'] if c['cell_type']=='code']; assert all(c['outputs'] for c in cells), 'code cells have no saved outputs'; print(f'OK: {len(cells)} code cells with outputs')"`
Expected: `OK: 6 code cells with outputs`

- [ ] **Step 4: Commit**

```bash
git add notebooks/07_Chat.ipynb
git commit -m "feat: add 07_Chat.ipynb interactive generation notebook"
```

---

### Task 4: README and full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full suite and linter to learn the test count**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`
Run: `.venv\Scripts\python.exe -m ruff check src/ tests/`
Expected: all tests pass (59 total after adding 12 new tests); ruff clean.

- [ ] **Step 2: Update README.md**

Edit the "Quick start" section, after the `ddm-train` block:

```markdown
# Chat with a trained model (streaming output in the terminal):
ddm-chat --checkpoint checkpoints/ddm_seed0.pt
```

Edit the notebook table to add a row at the end:

```markdown
| `07_Chat.ipynb` | Chat with trained models: comparisons, sampling sweep, segment-memory demo, interactive widget |
```

Edit the project layout line:

```markdown
src/ddm/       package: config, models, data, train, generate, viz, cli, chat
```

Update the test count sentence: `47 tests cover ...` → the number reported in Step 1, e.g. `59 tests cover ...`.

- [ ] **Step 3: Final verification and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: all pass

```bash
git add README.md
git commit -m "docs: document ddm-chat and 07_Chat.ipynb"
```
