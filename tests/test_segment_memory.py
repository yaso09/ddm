"""Segment memory: the memory token joins K/V and carries no gradient."""

from __future__ import annotations

import torch

from ddm.models import DDMModel, DistanceAwareAttention
from tests.conftest import BATCH, SEQ, VOCAB, make_config

T = 4
D = 16


def test_memory_token_appended_to_keys_and_values() -> None:
    """With a memory vector, attention runs over T + 1 key/value tokens."""
    torch.manual_seed(5)
    attn = DistanceAwareAttention(d_model=D, n_heads=2, learn_g=True, max_seq_len=32)
    x = torch.randn(1, T, D)
    memory = torch.randn(1, D)

    _, attn_no_mem = attn.get_attention_weights(x)
    _, attn_mem = attn.get_attention_weights(x, memory=memory)

    assert attn_no_mem.shape == (1, 2, T, T)
    assert attn_mem.shape == (1, 2, T, T + 1)


def test_memory_token_is_always_visible() -> None:
    """The causal mask must never mask out the memory column."""
    torch.manual_seed(6)
    attn = DistanceAwareAttention(d_model=D, n_heads=2, learn_g=True, max_seq_len=32)
    x = torch.randn(1, T, D)
    memory = torch.randn(1, D)

    _, attn_mem = attn.get_attention_weights(x, memory=memory)
    mem_column = attn_mem[..., 0]
    assert torch.isfinite(mem_column).all()
    assert (mem_column > 0).all(), "memory column must carry attention mass"


def test_memory_changes_output() -> None:
    """Different memories must produce different outputs."""
    torch.manual_seed(7)
    attn = DistanceAwareAttention(d_model=D, n_heads=2, learn_g=True, max_seq_len=32)
    x = torch.randn(1, T, D)
    out_a = attn(x, memory=torch.zeros(1, D))
    out_b = attn(x, memory=torch.ones(1, D) * 3.0)
    assert not torch.allclose(out_a, out_b, atol=1e-6)


def test_no_gradient_flows_into_memory_input() -> None:
    """memory is detached inside the attention: memory.grad stays None."""
    torch.manual_seed(8)
    attn = DistanceAwareAttention(d_model=D, n_heads=2, learn_g=True, max_seq_len=32)
    x = torch.randn(1, T, D)
    memory = torch.randn(1, D, requires_grad=True)

    loss = attn(x, memory=memory).sum()
    loss.backward()
    assert memory.grad is None, "gradient must not flow into the memory input"
    assert attn.q_proj.weight.grad is not None


def test_model_level_memory_is_detached() -> None:
    """new_memory is detached; full-model backward leaves it without grad."""
    torch.manual_seed(9)
    config = make_config("ddm", n_layers=1)
    model = DDMModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=1,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
        learn_g=True,
    )
    x = torch.randint(0, VOCAB, (BATCH, SEQ))
    memory = [torch.randn(BATCH, config.d_model, requires_grad=True)]

    logits, new_memory = model(x, memory=memory)
    assert not new_memory[0].requires_grad, "new_memory must be detached"
    assert new_memory[0].shape == (BATCH, config.d_model)

    loss = logits.sum()
    loss.backward()
    assert memory[0].grad is None, "no gradient may flow into the memory input"