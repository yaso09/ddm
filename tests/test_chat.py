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
    assert "[ddm-chat] context cleared" in out
    assert "tokens" in out


def test_unknown_checkpoint_exits_2(tmp_path, capsys) -> None:
    rc = main(["--checkpoint", str(tmp_path / "missing.pt"), "--device", "cpu"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "cannot load" in err