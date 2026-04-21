"""
PyTorch Dataset for diacritics restoration training.

Reads JSONL files with the schema produced by scripts/generate_dataset.py:

    {
        "input": <str, degraded text>,
        "target": <str, clean text>,
        "scenario": <str, one of strip_all|strip_partial|wrong_diacritics>,
        "changed_positions": [int, ...]
    }

Only `input` and `target` are used for seq2seq training; `scenario` is preserved
for per-scenario evaluation after training, and `changed_positions` is ignored
here (it belongs to the downstream detection-F1 eval, not training).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DiacriticsDataset(Dataset):
    """JSONL-backed dataset of (degraded, clean) Romanian text pairs."""

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        max_input_length: int = 1024,
        max_target_length: int = 1024,
        max_samples: int | None = None,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.examples: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))
                if max_samples is not None and len(self.examples) >= max_samples:
                    break

        logger.info("Loaded %d examples from %s", len(self.examples), self.path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]

        model_inputs = self.tokenizer(
            ex["input"],
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
        )
        labels = self.tokenizer(
            ex["target"],
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
        )["input_ids"]

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels,
        }

    # ------------------------------------------------------------------
    # Helpers used by the post-training per-scenario evaluator
    # ------------------------------------------------------------------

    def raw_triples(self) -> list[tuple[str, str, str]]:
        """Return (input, target, scenario) triples in order (no tokenization)."""
        return [
            (ex["input"], ex["target"], ex.get("scenario", "unknown"))
            for ex in self.examples
        ]
