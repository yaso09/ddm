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
        default="checkpoints/ddm_seed0.safetensors",
        help="Path to a checkpoint (default: checkpoints/ddm_seed0.safetensors).",
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