"""Data loading and tokenization for DDM experiments.

WikiText-2 (HuggingFace ``wikitext/wikitext-2-raw-v1``) is the default
corpus. Text is tokenized with the GPT-2 BPE tokenizer and split into
fixed-length blocks; the target for ``logits[:, t]`` is ``x[:, t + 1]``
(standard causal LM setup).
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def _hf_parquet_urls(repo_id: str, config: str, split: str) -> list[str]:
    """Query HF's public parquet-conversion API for a split's file URLs.

    This is the same endpoint the Hub's own dataset viewer uses
    (``https://huggingface.co/api/datasets/{repo_id}/parquet/{config}/{split}``)
    and works for any dataset that has an auto-converted parquet mirror,
    regardless of whether the original repo ships a loading script. Large
    splits are returned as multiple part files.

    Args:
        repo_id: Dataset repo id, e.g. ``agentlans/li2017dailydialog``.
        config: Dataset config/subset name (``"default"`` if the dataset has
            none of its own).
        split: Split name (``train`` / ``validation`` / ``test``).

    Returns:
        List of parquet file URLs for that split (usually one).
    """
    api_url = f"https://huggingface.co/api/datasets/{repo_id}/parquet/{config}/{split}"
    try:
        with urllib.request.urlopen(api_url, timeout=60) as resp:
            return json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(
            f"[data] parquet API unavailable ({type(exc).__name__}: {exc}); "
            "using auto-conversion resolve URLs..."
        )
        base = (
            f"https://huggingface.co/datasets/{repo_id}/resolve/"
            f"refs%2Fconvert%2Fparquet/{config}/{split}"
        )
        return [f"{base}/{i:04d}.parquet" for i in range(8)]


def _download_parquet(url: str, dest: Path) -> Path | None:
    """Download a parquet file if it is not cached yet.

    Args:
        url: HTTPS URL of the parquet file.
        dest: Local destination path.

    Returns:
        The local file path, or ``None`` when the URL does not exist
        (e.g. a missing auto-conversion part).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except (urllib.error.HTTPError, urllib.error.URLError):
        tmp.unlink(missing_ok=True)
        return None
    tmp.rename(dest)
    return dest


# Datasets whose useful content is a list-of-turns field rather than a flat
# ``text`` column. Each entry maps dataset_name -> the field holding the list
# of turns (see ``_format_dialog``); a turn is either a plain string
# (alternating User/Assistant) or a ``{"from": ..., "value": ...}`` dict
# (ShareGPT-style roles).
DIALOG_DATASETS: dict[str, str] = {
    # ShareGPT-format mirror of DailyDialog: parquet-native, no loading
    # script, so it doesn't hit the "scripts no longer supported" wall that
    # the canonical li2017dailydialog/daily_dialog repo does.
    "agentlans/li2017dailydialog": "conversations",
}

SPEAKER_TAGS: tuple[str, str] = ("User", "Assistant")
ROLE_TAGS: dict[str, str] = {
    "human": "User",
    "user": "User",
    "gpt": "Assistant",
    "assistant": "Assistant",
    # "system" is intentionally omitted -> those turns are dropped.
}


def _format_dialog(turns: list[Any]) -> str:
    """Render a list of turns as a User/Assistant transcript.

    Args:
        turns: Either plain strings (alternating speakers, starting with the
            user) or ``{"from": role, "value": text}`` dicts (ShareGPT-style;
            ``system`` turns are dropped, unrecognized roles are dropped).

    Returns:
        A single string with one ``"User: ..."`` / ``"Assistant: ..."`` line
        per turn, empty utterances dropped.
    """
    lines: list[str] = []
    for i, turn in enumerate(turns):
        if isinstance(turn, dict):
            tag = ROLE_TAGS.get(str(turn.get("from", "")).lower())
            text = str(turn.get("value", "")).strip()
        else:
            tag = SPEAKER_TAGS[i % 2]
            text = str(turn).strip()
        if tag and text:
            lines.append(f"{tag}: {text}")
    return "\n".join(lines)


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
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
        return load_dataset(dataset_name, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001 - fall back for any loader issue
        print(
            f"[data] script-based loading failed ({type(exc).__name__}: {exc}); "
            "switching to HF parquet API fallback..."
        )
        config_for_api = dataset_config or "default"
        safe_name = dataset_name.replace("/", "__")
        data_files: dict[str, list[str]] = {}
        for split in ("train", "validation", "test"):
            urls = _hf_parquet_urls(dataset_name, config_for_api, split)
            local_paths = []
            for i, url in enumerate(urls):
                dest = Path(cache_dir) / f"{safe_name}_{config_for_api}_{split}_{i}.parquet"
                local = _download_parquet(url, dest)
                if local is not None:
                    local_paths.append(str(local))
            data_files[split] = local_paths
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
        if dataset_name in DIALOG_DATASETS:
            field = DIALOG_DATASETS[dataset_name]
            texts = [
                _format_dialog(turns) for turns in ds[split_name][field] if turns
            ]
            # Blank line between conversations so the model can learn where
            # one dialogue ends and the next begins.
            joined = "\n\n".join(t for t in texts if t)
        else:
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