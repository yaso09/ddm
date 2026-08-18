# DDM — Distance-Decomposed Model

A causal language model whose attention weights are decomposed into a
content term and a *learned distance term*. A scalar gate
$g(k) \in (0,1)$, implemented as a two-layer MLP with a sigmoid output, is
applied before the softmax (in log-space); by the chain rule for
logarithms, each attention weight factorizes into content × distance. A
per-layer **segment memory** (detached mean hidden state of the previous
block, prepended as one key/value token) gives constant-time access to the
past regardless of context length.

See [`paper/paper.md`](paper/paper.md) for the full description and
`paper/paper.pdf` for the compiled document.

## Installation

```bash
python -m pip install -e ".[dev]"
```

(With a `uv`-managed venv: `uv pip install -e ".[dev]"`.)

## Quick start

```bash
# Train DDM on WikiText-2 and write results to checkpoints/
ddm-train --config configs/ddm_base.yaml

# Or, to keep a run short:
ddm-train --config configs/ddm_base.yaml --max-steps 1000 --epochs 1
```

Every config is a YAML file under `configs/`; see
[`src/ddm/config.py`](src/ddm/config.py) for all fields.

## Reproducing the experiments

The six notebooks under `notebooks/` reproduce the whole study end to end:

| Notebook | Contents |
|---|---|
| `01_Theory.ipynb` | The chain rule, Markov assumption, and the distance decomposition |
| `02_Implementation.ipynb` | DDM internals: gate MLP, ALiBi, segment memory, parameter accounting |
| `03_Training.ipynb` | Train DDM, save checkpoints, plot the learned gate curves |
| `04_Benchmark.ipynb` | Bigram vs 3-gram vs DDM vs Transformer (matched params), position buckets |
| `05_Ablation.ipynb` | Learned gate vs frozen $1/k$ gate; head-wise Welch $t$-tests |
| `06_Scaling.ipynb` | PPL / parameters / wall time across three model sizes |

Results tables are written to `checkpoints/benchmark_results.md` and
`checkpoints/scaling_results.md`.

## Tests

```bash
python -m pytest tests/ -v
ruff check src/ tests/
```

47 tests cover tensor shapes, causality, the pre-softmax gate, segment
memory, and ablation equivalence (model coverage ≥ 90%).

## Project layout

```
configs/       experiment YAML configs
notebooks/     01-06 executable analysis notebooks
paper/         paper.md, paper.pdf, build.sh / build.py
src/ddm/       package: config, models, data, train, viz, cli
tests/         pytest suite
checkpoints/   run artifacts (weights/results, git-ignored)
```

## Paper build

```bash
cd paper && ./build.sh        # uses pandoc+LaTeX when available
# fallback (no system deps):
python paper/build.py         # pip install pypandoc_binary typst
```

## License

MIT — see [LICENSE](LICENSE).