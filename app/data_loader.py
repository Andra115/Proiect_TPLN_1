"""
Data loading utilities.
Supports:
  - Raw text files / plain strings
  - HuggingFace datasets (RoLargeSum, RoMemes) — lazy import so the server
    starts even without `datasets` installed
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Minimum / maximum sentence length (characters) to keep
MIN_LEN = 20
MAX_LEN = 512


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Very simple sentence splitter — good enough for our pipeline."""
    sentences = _SENT_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if MIN_LEN <= len(s.strip()) <= MAX_LEN]


# ---------------------------------------------------------------------------
# Plain-text file loader
# ---------------------------------------------------------------------------

def load_text_file(path: str | Path, max_sentences: int | None = None) -> list[str]:
    """Load a plain text file and return clean sentences."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    sentences = split_sentences(text)
    if max_sentences:
        sentences = sentences[:max_sentences]
    logger.info("Loaded %d sentences from %s", len(sentences), path)
    return sentences


# ---------------------------------------------------------------------------
# HuggingFace dataset loaders
# ---------------------------------------------------------------------------

def _require_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Install `datasets` to load HuggingFace data: pip install datasets"
        ) from e


def load_rolargesum(
    split: str = "train",
    max_sentences: int = 10_000,
    streaming: bool = True,
) -> list[str]:
    """
    Load Romanian news sentences from RoLargeSum.
    Repo: https://github.com/avramandrei/rolargesum
    HuggingFace name: 'avramandrei/RoLargeSum'
    """
    _require_datasets()
    from datasets import load_dataset  # type: ignore

    logger.info("Downloading RoLargeSum (%s)…", split)
    ds = load_dataset(
        "avramandrei/RoLargeSum",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )

    sentences: list[str] = []
    for row in ds:
        # dataset has 'article' and 'summary' columns
        for col in ("article", "summary"):
            if col in row and row[col]:
                sentences.extend(split_sentences(row[col]))
        if max_sentences and len(sentences) >= max_sentences:
            break

    sentences = sentences[:max_sentences]
    logger.info("Loaded %d sentences from RoLargeSum", len(sentences))
    return sentences

# ---------------------------------------------------------------------------
# Quick demo / test data (no downloads needed)
# ---------------------------------------------------------------------------

DEMO_SENTENCES = [
    "Câmpul de grâu se întindea până la marginea orizontului.",
    "Ștefan cel Mare a fost un domnitor important al Moldovei.",
    "Mâine dimineață vom pleca la munte împreună cu familia.",
    "Elevii au învățat despre istoria României în cadrul orei.",
    "Ploaia torențială a inundat străzile orașului în câteva ore.",
    "Îngrijitorii de la grădina zoologică hrănesc animalele în fiecare zi.",
    "Codul sursă al aplicației trebuie să fie bine documentat.",
    "Orașul Iași este cunoscut pentru universitatea sa prestigioasă.",
    "Fântâna din centrul parcului a fost renovată recent.",
    "Echipa de fotbal a câștigat campionatul după ani de eforturi.",
]