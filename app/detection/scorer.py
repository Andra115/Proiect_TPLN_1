"""
detection/scorer.py

Computes a real uncertainty score for each detected character change using
a character-level n-gram language model trained on Romanian reference text.

Why n-grams and not just a lookup table?
----------------------------------------
The project spec (§3.3) asks for "probabilitate/entropie pe caracter" — an
actual probability estimate, not a hard-coded weight.  Since we do not have
access to the seq2seq model's internal softmax distributions (§3.2 is not
integrated yet), we approximate P(correct_char | left_context) with a
trigram/bigram character LM with Laplace smoothing, built from reference text.

This gives us a real conditional probability we can convert to entropy, which
is a principled stand-in until real model logits are available.  The swap is
trivial: replace `score_change()` with one that reads logits from the model.

Usage
-----
    scorer = DiacriticScorer()
    scorer.fit(reference_texts)          # call once at startup
    score = scorer.score_change(change)  # returns UncertaintyScore
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .diff import Change


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyScore:
    probability: float    # P(corrected_char | left_context),  0 < p ≤ 1
    entropy: float        # Shannon entropy H over the confusion set, in bits
    flag_for_review: bool # True when uncertainty exceeds threshold
    method: str           # 'ngram' | 'fallback'


# ---------------------------------------------------------------------------
# Confusion sets — characters that can plausibly appear at the same position
# These define the distribution we compute entropy over.
# ---------------------------------------------------------------------------
_CONFUSION_SETS: dict[str, list[str]] = {
    "ș": ["s", "ș", "ş"],
    "s": ["s", "ș", "ş"],
    "ş": ["s", "ș", "ş"],
    "ț": ["t", "ț", "ţ"],
    "t": ["t", "ț", "ţ"],
    "ţ": ["t", "ț", "ţ"],
    "ă": ["a", "ă", "â"],
    "â": ["a", "ă", "â"],
    "a": ["a", "ă", "â"],
    "î": ["i", "î"],
    "i": ["i", "î"],
}

# Flag when entropy exceeds this many bits (tune as needed)
_ENTROPY_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# N-gram character language model
# ---------------------------------------------------------------------------

class CharNgramLM:
    """
    Bidirectional character n-gram LM.
    Combines a forward trigram P(c | left_context) with a backward trigram
    P(c | right_context) — right context is fed in reverse, so the
    immediate right neighbour is always the first character seen.
    """

    def __init__(self, order: int = 5):
        self.order = order
        # Forward: P(c | prev chars)
        self._fwd: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._fwd_totals: dict[str, int] = defaultdict(int)
        # Backward: P(c | next chars, reversed)
        self._bwd: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._bwd_totals: dict[str, int] = defaultdict(int)
        self._vocab: set[str] = set()

    def fit(self, texts: Sequence[str]) -> None:
        for text in texts:
            t = text.lower()
            pad = "\x00" * (self.order - 1)
            padded_fwd = pad + t
            padded_bwd = pad + t[::-1]  # reversed for backward pass

            for i in range(self.order - 1, len(padded_fwd)):
                ctx = padded_fwd[i - (self.order - 1): i]
                char = padded_fwd[i]
                self._fwd[ctx][char] += 1
                self._fwd_totals[ctx] += 1
                self._vocab.add(char)

            for i in range(self.order - 1, len(padded_bwd)):
                ctx = padded_bwd[i - (self.order - 1): i]
                char = padded_bwd[i]
                self._bwd[ctx][char] += 1
                self._bwd_totals[ctx] += 1

    def prob(self, char: str, left_context: str, right_context: str = "") -> float:
        """
        Combined P(char | left, right) using geometric mean of forward
        and backward probabilities.
        Geometric mean keeps both signals balanced — if either is very
        confident, it pulls the combined score strongly.
        """
        if not self._vocab:
            return 1.0

        vocab_size = max(len(self._vocab), 1)
        char_l = char.lower()

        # Adjusted Smoothing: Use a smaller fractional factor instead of Add-1
        # This keeps the LM opinionated and prevents baseline entropy from spiking.
        alpha = 0.05

        # Forward: P(c | left)
        fwd_ctx = left_context[-(self.order - 1):].lower()
        fwd_ctx = fwd_ctx.zfill(self.order - 1)
        p_fwd = (self._fwd[fwd_ctx][char_l] + alpha) / (self._fwd_totals[fwd_ctx] + vocab_size * alpha)

        # Backward: P(c | right) — right context reversed, immediate neighbour first
        if right_context:
            bwd_ctx = right_context[: self.order - 1][::-1].lower()
            bwd_ctx = bwd_ctx.zfill(self.order - 1)
            p_bwd = (self._bwd[bwd_ctx][char_l] + alpha) / (self._bwd_totals[bwd_ctx] + vocab_size * alpha)
        else:
            p_bwd = p_fwd  # no right context available — don't penalise

        # Geometric mean
        return math.sqrt(p_fwd * p_bwd)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class DiacriticScorer:
    """
    Wraps CharNgramLM to produce UncertaintyScore for each Change.

    Call `fit()` once at app startup with your reference corpus.
    Then call `score_change()` per detected change.
    """

    def __init__(self, review_threshold: float = _ENTROPY_THRESHOLD):
        self._lm = CharNgramLM(order=5)
        self._fitted = False
        self._threshold = review_threshold

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, texts: Sequence[str]) -> None:
        """
        Train the n-gram LM on reference Romanian text.
        `texts` can be sentences or longer paragraphs — the LM sees them
        all as a stream of characters.
        """
        self._lm.fit(texts)
        self._fitted = True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_change(self, change: Change, left_context: str = "", right_context: str = "") -> UncertaintyScore:
        target_char = change.corrected_char
        if not target_char:
            return UncertaintyScore(probability=0.99, entropy=0.0,
                                    flag_for_review=False, method="fallback")

        confusion_set = _CONFUSION_SETS.get(target_char.lower())
        if confusion_set is None or not self._fitted:
            return UncertaintyScore(probability=0.95, entropy=0.0,
                                    flag_for_review=False, method="fallback")

        raw_probs = {
            c: self._lm.prob(c, left_context, right_context)
            for c in confusion_set
        }
        total = sum(raw_probs.values())
        norm_probs = {c: p / total for c, p in raw_probs.items()}

        p_correct = norm_probs.get(target_char.lower(), 1e-6)
        entropy = -sum(
            p * math.log2(p + 1e-9)
            for p in norm_probs.values()
            if p > 0
        )

        # Flagging Logic: Catch ambiguity (high entropy) OR blatant errors (low probability)
        is_ambiguous = entropy >= self._threshold
        is_blatant_error = p_correct < 0.15

        return UncertaintyScore(
            probability=round(p_correct, 6),
            entropy=round(entropy, 6),
            flag_for_review=bool(is_ambiguous or is_blatant_error),
            method="ngram",
        )

    def score_all(self, changes: list[Change], output_text: str) -> list[UncertaintyScore]:
        scores: list[UncertaintyScore] = []
        for ch in changes:
            left_ctx = output_text[: ch.position]
            right_ctx = output_text[ch.position + len(ch.corrected_char):]  # everything after
            scores.append(self.score_change(ch, left_context=left_ctx, right_context=right_ctx))
        return scores