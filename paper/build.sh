#!/usr/bin/env bash
# Build paper.pdf from paper.md.
#
# Uses pandoc with a LaTeX engine (pdflatex/xelatex) when available, falls
# back to the bundled pandoc-binary + typst pipeline in build.py otherwise.
set -euo pipefail
cd "$(dirname "$0")"

if command -v pandoc >/dev/null 2>&1; then
    if command -v pdflatex >/dev/null 2>&1; then
        pandoc paper.md -o paper.pdf --pdf-engine=pdflatex
    elif command -v xelatex >/dev/null 2>&1; then
        pandoc paper.md -o paper.pdf --pdf-engine=xelatex
    elif command -v typst >/dev/null 2>&1; then
        pandoc paper.md -o paper.pdf --pdf-engine=typst
    else
        echo "[paper] pandoc found but no LaTeX engine; using bundled build..." >&2
        python build.py
    fi
else
    echo "[paper] pandoc not found; using bundled build (pypandoc + typst)..." >&2
    python build.py
fi
echo "[paper] done: paper/paper.pdf"