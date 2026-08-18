"""Data loading and tokenization for DDM experiments.

WikiText-2 (HuggingFace ``wikitext/wikitext-2-raw-v1``) is the default
corpus. Text is tokenized with the GPT-2 BPE tokenizer and split into
fixed-length blocks; the target for ``logits[:, t]`` is ``x[:, t + 1]``
(standard causal LM setup).
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

DATASET_URL = (
    "https://huggingface.co/datasets/{name}/resolve/main/"
    "{config}/{split}-00000-of-00001.parquet"
)


def _download_parquet(url: str, dest: Path) -> Path:
    """Download a parquet file if it is not cached yet.

    Args:
        url: HTTPS URL of the parquet file.
        dest: Local destination path.

    Returns:
        The local file path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.rename(dest)
    return dest


def _load_dataset(
    dataset_name: str,
    dataset_config: str,
    cache_dir: str,
) -> Any:
    """Load a HuggingFace dataset with a local-parquet fallback.

    Newer ``datasets`` releases cannot access namespaceless repositories
    (e.g. ``wikitext``): the script-based loader raises ``HfUriError`` and
    even plain-parquet ``data_files`` URLs are rewritten to ``hf://`` URIs,
    which fails for the same reason. The fallback therefore downloads the
    parquet snapshots over plain HTTPS with ``urllib`` and loads them from
    local paths (byte-identical content).

    Args:
        dataset_name: Dataset name, e.g. ``wikitext``.
        dataset_config: Dataset configuration, e.g. ``wikitext-2-raw-v1``.
        cache_dir: Directory to cache the downloaded data.

    Returns:
        The loaded dataset (``DatasetDict``-like object).
    """
    try:
        return load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001 - fall back for any loader issue
        print(
            f"[data] script-based loading failed ({type(exc).__name__}: {exc}); "
            "switching to local parquet fallback..."
        )
        data_files: dict[str, str] = {}
        for split in ("train", "validation", "test"):
            url = DATASET_URL.format(
                name=dataset_name, config=dataset_config, split=split
            )
            dest = Path(cache_dir) / f"{dataset_config}_{split}.parquet"
            data_files[split] = str(_download_parquet(url, dest))
        return load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)


def _get_tokenizer() -> Any:
    """Return the GPT-2 BPE tokenizer.

    Returns:
        A tokenizer with ``pad_token`` set to the EOS token.
    """
    try:
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except ImportError:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_and_tokenize(
    seq_len: int = 128,
    cache_dir: str = "hf_cache",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    max_blocks: int | None = None,
) -> tuple[dict[str, torch.Tensor], int, Any]:
    """Download, tokenize and chunk a text corpus.

    Args:
        seq_len: Fixed block length.
        cache_dir: Directory for the dataset cache.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration.
        max_blocks: If set, keep only the first ``max_blocks`` blocks of each
            split (useful for smoke runs).

    Returns:
        Tuple ``(blocks, vocab_size, tokenizer)`` where ``blocks`` maps split
        names to ``[n_blocks, seq_len]`` tensors, ``vocab_size`` is the
        tokenizer vocabulary size and ``tokenizer`` the tokenizer itself.
    """
    print(f"[data] loading {dataset_name}/{dataset_config} (cache: {cache_dir})...")
    ds = _load_dataset(dataset_name, dataset_config, cache_dir)
    tokenizer = _get_tokenizer()

    def tokenize_split(split_name: str) -> torch.Tensor:
        texts = [t for t in ds[split_name]["text"] if t.strip() != ""]
        joined = "\n".join(texts)
        ids = tokenizer(joined, return_tensors="pt")["input_ids"][0]
        n_blocks = len(ids) // seq_len
        blocks = ids[: n_blocks * seq_len].view(n_blocks, seq_len)
        if max_blocks is not None:
            blocks = blocks[:max_blocks]
        return blocks

    out = {
        "train": tokenize_split("train"),
        "val": tokenize_split("validation"),
        "test": tokenize_split("test"),
    }
    print(
        f"[data] blocks -> train: {tuple(out['train'].shape)}, "
        f"val: {tuple(out['val'].shape)}, test: {tuple(out['test'].shape)}"
    )
    return out, tokenizer.vocab_size, tokenizer


class BlockDataset(Dataset):
    """Dataset of (input, target) pairs from fixed-length blocks.

    Each block of length ``L`` yields input ``seq[:-1]`` and target
    ``seq[1:]`` (both of length ``L - 1``), i.e. every position predicts the
    next token.

    Args:
        blocks: Token blocks of shape ``[n_blocks, seq_len]``.
    """

    def __init__(self, blocks: torch.Tensor) -> None:
        self.blocks = blocks

    def __len__(self) -> int:
        """Number of blocks."""
        return self.blocks.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (input, target) pair for a block.

        Args:
            idx: Block index.

        Returns:
            Tuple of input ids and target ids, each of length ``seq_len - 1``.
        """
        seq = self.blocks[idx]
        return seq[:-1], seq[1:]


def make_loader(
    blocks: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a DataLoader over a split's blocks.

    Args:
        blocks: Token blocks ``[n_blocks, seq_len]``.
        batch_size: Batch size (last partial batch is dropped).
        shuffle: Whether to shuffle the blocks.

    Returns:
        A ``DataLoader`` yielding ``(x, y)`` pairs of shape
        ``[B, seq_len - 1]``.
    """
    return DataLoader(
        BlockDataset(blocks),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
