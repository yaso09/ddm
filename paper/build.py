"""Bundled paper build: pandoc (via pypandoc-binary) -> typst -> PDF.

Used when ``pandoc``/LaTeX is not available on PATH. Install the
dependencies with::

    pip install pypandoc_binary typst

Usage::

    python build.py [input.md] [output.pdf]

"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper.md"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper.pdf"
    typ = ROOT / (dst.stem + ".typ")
    try:
        import pypandoc
        import typst
    except ImportError as exc:
        print(
            f"[paper] missing dependency ({exc}); install with: "
            "pip install pypandoc_binary typst"
        )
        return 1
    pypandoc.convert_file(str(src), "typst", outputfile=str(typ))
    typst.compile(str(typ), output=str(dst))
    print(f"[paper] wrote {dst} ({dst.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())