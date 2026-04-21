"""
Inference helper for diacritics restoration.

Loads a fine-tuned ByT5 checkpoint and exposes a `restore` method that returns
the restored text, the character positions that changed, and an overall
confidence score (mean token probability).

This module is what `app/main.py` (the FastAPI service) should import once
a trained model exists at `models/byt5-diacritics/final`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class RestorationResult:
    input: str
    restored: str
    changed_positions: list[int]   # indices in `restored`
    confidence: float              # mean per-token probability, in [0, 1]

    def to_dict(self) -> dict:
        return asdict(self)


class DiacriticsRestorer:
    """Wrapper around a fine-tuned ByT5 model for diacritics restoration."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        max_input_length: int = 1024,
        max_output_length: int = 1024,
        num_beams: int = 4,
    ):
        self.model_path = str(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams

        logger.info("Loading model from %s on %s", self.model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def restore(self, text: str) -> RestorationResult:
        if not text.strip():
            return RestorationResult(
                input=text, restored=text, changed_positions=[], confidence=1.0
            )

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_output_length,
            num_beams=self.num_beams,
            return_dict_in_generate=True,
            output_scores=True,
        )

        restored = self.tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        confidence = self._compute_confidence(out)
        changed_positions = self._diff_positions(text, restored)

        return RestorationResult(
            input=text,
            restored=restored,
            changed_positions=changed_positions,
            confidence=confidence,
        )

    def restore_batch(self, texts: list[str]) -> list[RestorationResult]:
        """Simple per-item loop; adequate for demo throughput. Swap for a batched
        generate() if you need high QPS."""
        return [self.restore(t) for t in texts]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _diff_positions(original: str, restored: str) -> list[int]:
        """Character positions in `restored` that differ from `original`.

        Fast path for equal-length strings; Levenshtein alignment otherwise.
        """
        if len(original) == len(restored):
            return [i for i, (a, b) in enumerate(zip(original, restored)) if a != b]
        return DiacriticsRestorer._diff_positions_align(original, restored)

    @staticmethod
    def _diff_positions_align(original: str, restored: str) -> list[int]:
        m, n = len(original), len(restored)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if original[i - 1] == restored[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]
                    )

        changed: list[int] = []
        i, j = m, n
        while i > 0 and j > 0:
            if original[i - 1] == restored[j - 1]:
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:   # substitution
                changed.append(j - 1)
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i][j - 1] + 1:       # insertion into restored
                changed.append(j - 1)
                j -= 1
            else:                                     # deletion from original
                i -= 1
        while j > 0:
            changed.append(j - 1)
            j -= 1
        return sorted(changed)

    def _compute_confidence(self, generation_output) -> float:
        """Mean token probability of the chosen sequence.

        With beam search, HF exposes `sequences_scores` (length-normalized log-prob
        of the top beam) — we exponentiate that. For greedy, we sum over the
        per-step scores.
        """
        seq_scores = getattr(generation_output, "sequences_scores", None)
        if seq_scores is not None:
            log_prob = float(seq_scores[0].item())
            return float(math.exp(max(log_prob, -20.0)))

        scores = getattr(generation_output, "scores", None)
        if not scores:
            return 1.0

        seq = generation_output.sequences[0]
        probs: list[float] = []
        for step, logits in enumerate(scores):
            step_probs = torch.softmax(logits[0], dim=-1)
            tok = int(seq[step + 1].item())  # +1 skips the decoder-start token
            if tok == self.tokenizer.eos_token_id:
                break
            probs.append(float(step_probs[tok].item()))
        return float(sum(probs) / len(probs)) if probs else 1.0
