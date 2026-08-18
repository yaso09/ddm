"""Command-line interface: ``ddm-train``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ddm.config import DDMConfig
from ddm.train import format_results_table, run_training


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="ddm-train",
        description="Train a DDM experiment defined by a YAML config.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML config (see configs/).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap training steps per epoch (overrides config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ddm-train`` console script.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 on success).
    """
    args = build_parser().parse_args(argv)
    config = DDMConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.epochs is not None:
        config.epochs = args.epochs

    results = run_training(config)
    table = format_results_table(results)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    table_path = save_dir / "results.md"
    table_path.write_text(table, encoding="utf-8")
    print(f"\n[cli] results table written to {table_path}")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
