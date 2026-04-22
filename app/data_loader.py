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

# ---------------------------------------------------------------------------
# Local JSONL loader (RoLargeSum pre-downloaded)
# ---------------------------------------------------------------------------

def load_rolargesum_local(
    path: str | Path | None = None,
    max_sentences: int = 20_000,
) -> list[str]:
    """
    Load clean Romanian sentences from a local RoLargeSum JSONL file.
    Each line is a JSON object with at minimum a 'target' field containing
    correctly diacritized text (as produced by the degradation pipeline).

    Falls back to DEMO_SENTENCES if the file is not found.
    """
    if path is None:
        # Default: look for data/train.jsonl relative to project root
        path = Path(__file__).parent.parent / "data" / "train.jsonl"

    path = Path(path)
    if not path.exists():
        logger.warning("Local JSONL not found at %s — falling back to DEMO_SENTENCES", path)
        return DEMO_SENTENCES

    import json
    sentences: list[str] = []
    skipped = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # 'target' is the clean diacritized text
                text = obj.get("target") or obj.get("article") or obj.get("summary") or ""
                if text:
                    sentences.extend(split_sentences(text))
            except json.JSONDecodeError:
                skipped += 1
            if len(sentences) >= max_sentences:
                break

    if skipped:
        logger.warning("Skipped %d malformed lines in %s", skipped, path)

    sentences = sentences[:max_sentences]
    logger.info("Loaded %d sentences from local JSONL %s", len(sentences), path)
    return sentences