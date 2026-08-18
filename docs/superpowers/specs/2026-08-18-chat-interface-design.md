# Design: Interactive chat with trained models (CLI + notebook)

Date: 2026-08-18

## Goal

Let the user converse with any trained checkpoint from `checkpoints/`
interactively — from the terminal via a new `ddm-chat` command, and from
a notebook (`07_Chat.ipynb`) with demos plus an ipywidgets chat widget.

The models are WikiText-2 language models, so "chat" means prompt
continuation. Sessions are multi-turn: the conversation accumulates in
context and DDM's segment memory is carried across turns.

## Components

```
src/ddm/generate.py     core: load_checkpoint(), generate()
src/ddm/chat.py         REPL: ddm-chat entry point main()
pyproject.toml          [project.scripts] ddm-chat = "ddm.chat:main"
notebooks/07_Chat.ipynb demos + ipywidgets chat
tests/test_generate.py  core tests (CPU-forced)
tests/test_chat.py      CLI smoke tests (CPU-forced)
```

The CLI and the notebook share one code path: both call `generate()`.

## Generation core (`src/ddm/generate.py`)

### `load_checkpoint(path) -> (model, config, tokenizer)`

Read a checkpoint `{"state_dict": ..., "config": asdict(DDMConfig)}`
(as written by `run_training` in `src/ddm/train.py`), rebuild the model
via `build_model(config)`, load `state_dict`, and load the GPT-2 BPE
tokenizer (`GPT2TokenizerFast.from_pretrained("gpt2")`, same as
`_get_tokenizer` in `src/ddm/data.py`).

### `generate(...) -> GenerationResult`

Signature:

- `model`, `tokenizer`, `prompt: str`
- `max_new_tokens: int = 128`
- `temperature: float = 0.8`
- `top_k: int = 50` (0 = disabled)
- `top_p: float = 0.9` (1.0 = disabled)
- `seed: int | None = None` (deterministic when set)
- `device: str = ""` (auto-detect: CUDA if available, else CPU)
- `memory: list[torch.Tensor] | None = None` (DDM segment memory carried
  from the previous turn)
- `on_token: Callable[[str], None] | None = None` (streaming callback,
  called with each decoded token)

Returns `GenerationResult(text, token_ids, memory, tokens_per_sec)`.

Sampling pipeline per step: temperature scaling → top-k filter → top-p
(nucleus) filter → softmax → multinomial draw. Generation stops when
`max_new_tokens` is reached or the EOS token (`<|endoftext|>`) is
drawn; EOS is excluded from the returned text. Runs under
`@torch.no_grad()` with the model in `eval()` mode.

Per-model input window:

- `ddm` / `ddm_ablation`: last `max_seq_len` tokens + carried per-layer
  memory; each step produces new memory passed to the next step.
- `transformer`: last `max_seq_len` tokens, no memory.
- `bigram`: last 1 token; `ngram`: last `n_context` tokens. Simple
  models use the prompt only for the first step, then their own window.

## CLI (`src/ddm/chat.py`)

```
$ ddm-chat --checkpoint checkpoints/ddm_seed0.pt
ddm-chat: ddm (seed 0) — 2 layers, 4 heads, d_model 128 | type /help
> The capital of France is
<streaming output on one line>
> /reset
> /quit
```

Flags:

- `--checkpoint` (default `checkpoints/ddm_seed0.pt`)
- `--max-new-tokens` (default 128)
- `--temperature` (default 0.8)
- `--top-k` (default 50)
- `--top-p` (default 0.9)
- `--seed` (default None = random)
- `--device` (default "" = auto-detect: GPU if available)

REPL commands: `/help` (list), `/reset` (clear context + memory),
`/quit` or Ctrl-D/Ctrl-C (exit). Any other line is a prompt.

Output per turn: `> ` prompt, streamed generation, then a summary line
(`n tokens, X tok/s, wall time`); for DDM a small note that memory was
carried. Unknown commands print a short warning and the session
continues.

Errors: missing/corrupt checkpoint → explanatory message, exit code 2.

## Notebook (`notebooks/07_Chat.ipynb`)

Follows the style of the existing notebooks (markdown + code cells,
pre-executed outputs saved):

1. Intro: purpose, checkpoints, `generate()` API.
2. One-shot demos: same prompt across ddm / transformer / bigram /
   ngram — shows the simple models' weakness.
3. Sampling sweep: temperature and top-k/top-p variations with the same
   seed and prompt.
4. Long-context demonstration: 200+ token story, then a follow-up —
   DDM's segment-memory advantage vs transformer truncation at 128
   tokens.
5. Interactive chat widget: ipywidgets — model dropdown, `max_new_tokens`
   / `temperature` sliders, text box + run button, output area. Uses
   `generate()` with the `on_token` callback.
6. Summary: observation table (which model uses how much context).

## Device policy

- Application (CLI + notebook): auto-detect — CUDA if available, else
  CPU (`_pick_device` logic from `src/ddm/train.py`).
- Tests: always CPU (`device="cpu"` forced), fast small model configs.

## Tests

`tests/test_generate.py`:

- Determinism: same seed → identical output; different seed → generally
  different.
- `max_new_tokens` cap and EOS early stopping (token sequence ending in
  EOS).
- Temperature / top-k / top-p applied correctly to logits (small
  synthetic model).
- DDM memory carry: two-turn `generate()` with memory — output changes;
  passing `None` memory produces fresh memory.
- `load_checkpoint`: round-trip a synthetic checkpoint, verify model
  structure and config match.

`tests/test_chat.py`:

- REPL smoke test: `main([...])` fed prompt + `/reset` + `/quit` via
  stdin, output contains generated text.
- Unknown checkpoint → exit code 2.

All tests use small configs (`d_model=32`, `n_layers=1`) on CPU, in the
style of the existing 47 tests.

## Out of scope

- Real instruction-following / dialogue (models are WikiText-2 LMs).
- KV-cache based fast inference (segment memory already bounds cost).
- GPU-only features; CPU must always work.