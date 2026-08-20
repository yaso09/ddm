"""Generation core: sampling filters, EOS handling, DDM memory carry, checkpoint loading."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

import ddm.generate as gen
from ddm.models import DDMModel, build_model
from tests.conftest import make_config


class FakeTokenizer:
    """Offline stand-in for the GPT-2 tokenizer (27-token vocab, EOS = 26)."""

    eos_token_id = 26

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [min(ord(c) % 26, 25) for c in text] or [0]
        if return_tensors is None:
            return {"input_ids": ids}
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
    assert continued.memory is not None
    x = torch.tensor([[0, 1]])
    with torch.no_grad():
        logits_plain, _ = gen.forward_step(model, x, None)
        logits_with_mem, _ = gen.forward_step(model, x, first.memory)
    assert not torch.allclose(logits_with_mem, logits_plain, atol=1e-6)


def test_baseline_returns_no_memory() -> None:
    config = make_config("ngram", vocab_size=27, n_context=3)
    model = build_model(config)
    tok = FakeTokenizer()
    out = gen.generate(model, tok, "abc", max_new_tokens=5, temperature=0, device="cpu")
    assert out.memory is None


def test_load_checkpoint_roundtrip(tmp_path, monkeypatch) -> None:
    config = make_config("bigram", vocab_size=27)
    model = build_model(config)
    path = tmp_path / "bigram_seed0.safetensors"
    save_file(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        path,
        metadata={"config": json.dumps(asdict(config))},
    )
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