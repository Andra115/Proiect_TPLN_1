import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# Diacritic mappings

DIACRITIC_TO_BASE: dict[str, str] = {
    "ș": "s", "Ș": "S",
    "ț": "t", "Ț": "T",
    "ă": "a", "Ă": "A",
    "â": "a", "Â": "A",
    "î": "i", "Î": "I",
}

CEDILLA_TO_CORRECT: dict[str, str] = {
    "ş": "ș", "Ş": "Ș",
    "ţ": "ț", "Ţ": "Ț",
}

BASE_TO_DIACRITICS: dict[str, list[str]] = {
    "s": ["ș"], "S": ["Ș"],
    "t": ["ț"], "T": ["Ț"],
    "a": ["ă", "â"], "A": ["Ă", "Â"],
    "i": ["î"], "I": ["Î"],
}

OCR_CONFUSIONS: list[tuple[str, str]] = [
    ("1", "l"), ("l", "1"),
    ("0", "o"), ("o", "0"),
    ("rn", "m"), ("m", "rn"),
    ("ii", "u"), ("u", "ii"),
    ("cl", "d"), ("vv", "w"),
]


# Scenario types

class DegradationScenario(str, Enum):
    STRIP_ALL = "strip_all"           # Remove all diacritics
    STRIP_PARTIAL = "strip_partial"   # Remove some diacritics randomly
    WRONG_DIACRITICS = "wrong_diacritics"  # Swap diacritics (e.g. â→î, ă→a then back wrong)
    CEDILLA = "cedilla"               # Replace correct diacritics with cedilla variants
    OCR = "ocr"                       # OCR-style character-level noise
    MIXED = "mixed"                   # Combination of above


@dataclass
class DegradationConfig:
    scenario: DegradationScenario = DegradationScenario.MIXED

    # Strip-partial params
    strip_probability: float = 0.5    # Per-character probability of stripping when partial

    # Wrong-diacritics params
    wrong_diacritic_probability: float = 0.3   # P(introduce wrong diacritic on base char)
    wrong_swap_probability: float = 0.2        # P(swap existing diacritic with wrong one)

    # OCR params
    ocr_probability: float = 0.02     # Per-token probability of OCR substitution
    ocr_insert_space_prob: float = 0.01  # P(random space insertion / merge)

    # Mixed weights (must sum to 1)
    mixed_weights: list[float] = field(
        default_factory=lambda: [0.25, 0.25, 0.2, 0.1, 0.1, 0.1]
        # strip_all, strip_partial, wrong_diacritics, cedilla, ocr, combined
    )

    seed: Optional[int] = None


# Core degradation functions

def strip_all_diacritics(text: str) -> str:
    """Remove every Romanian diacritic, replacing with base character."""
    for diac, base in DIACRITIC_TO_BASE.items():
        text = text.replace(diac, base)
    return text


def strip_partial_diacritics(text: str, probability: float, rng: random.Random) -> str:
    """Remove each diacritic independently with the given probability."""
    result = []
    for ch in text:
        if ch in DIACRITIC_TO_BASE and rng.random() < probability:
            result.append(DIACRITIC_TO_BASE[ch])
        else:
            result.append(ch)
    return "".join(result)


def introduce_wrong_diacritics(
    text: str,
    wrong_diacritic_prob: float,
    wrong_swap_prob: float,
    rng: random.Random,
) -> str:
    """
    Two operations:
    1. On base chars that *could* have a diacritic, randomly introduce the wrong one.
    2. On chars that already have a diacritic, randomly swap to a different diacritic.
    """
    # First strip all so we work on a clean base
    stripped = strip_all_diacritics(text)

    result = []
    for ch in stripped:
        candidates = BASE_TO_DIACRITICS.get(ch)
        if candidates and rng.random() < wrong_diacritic_prob:
            # Pick any of the possible diacritics (could be "right" or "wrong" — that's the point)
            result.append(rng.choice(candidates))
        else:
            result.append(ch)
    degraded = "".join(result)

    # Second pass: swap some existing diacritics to a sibling form
    # e.g. ș→ț would be too wild; we do things like â↔î, ă↔â
    swap_groups = [
        ["â", "î"],   # both map from a/i
        ["ă", "â"],
        ["ș", "s"],
        ["ț", "t"],
    ]
    final = []
    for ch in degraded:
        swapped = False
        for group in swap_groups:
            if ch in group and rng.random() < wrong_swap_prob:
                others = [x for x in group if x != ch]
                final.append(rng.choice(others))
                swapped = True
                break
        if not swapped:
            final.append(ch)
    return "".join(final)


def apply_cedilla_variants(text: str, rng: random.Random, probability: float = 0.7) -> str:
    """Replace correct ș/ț with cedilla variants ş/ţ (common in legacy Romanian text)."""
    result = []
    reverse_map = {v: k for k, v in CEDILLA_TO_CORRECT.items()}
    for ch in text:
        if ch in reverse_map and rng.random() < probability:
            result.append(reverse_map[ch])
        else:
            result.append(ch)
    return "".join(result)


def apply_ocr_noise(text: str, ocr_prob: float, space_prob: float, rng: random.Random) -> str:
    """Apply OCR-style substitutions and split/merge errors."""
    # Character-level substitutions
    for original, replacement in OCR_CONFUSIONS:
        if rng.random() < ocr_prob:
            text = text.replace(original, replacement)

    # Random space insertion (split) or removal (merge)
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        result.append(ch)
        if ch == " " and rng.random() < space_prob:
            pass  # drop the space (merge)
        elif ch != " " and rng.random() < space_prob / 2:
            result.append(" ")  # insert spurious space
        i += 1
    return "".join(result)


# Main degradation entry point

_SCENARIO_LIST = [
    DegradationScenario.STRIP_ALL,
    DegradationScenario.STRIP_PARTIAL,
    DegradationScenario.WRONG_DIACRITICS,
    DegradationScenario.CEDILLA,
    DegradationScenario.OCR,
]


def degrade_text(text: str, config: DegradationConfig) -> tuple[str, DegradationScenario]:
    """
    Apply degradation to a single text string.
    Returns (degraded_text, scenario_used).
    """
    rng = random.Random(config.seed)
    degraded = _apply_single(text, config.scenario, config, rng)
    return degraded, config.scenario


def _apply_single(
    text: str, scenario: DegradationScenario, config: DegradationConfig, rng: random.Random
) -> str:
    if scenario == DegradationScenario.STRIP_ALL:
        return strip_all_diacritics(text)
    elif scenario == DegradationScenario.STRIP_PARTIAL:
        return strip_partial_diacritics(text, config.strip_probability, rng)
    elif scenario == DegradationScenario.WRONG_DIACRITICS:
        return introduce_wrong_diacritics(
            text, config.wrong_diacritic_probability, config.wrong_swap_probability, rng
        )
    elif scenario == DegradationScenario.CEDILLA:
        return apply_cedilla_variants(text, rng)
    elif scenario == DegradationScenario.OCR:
        return apply_ocr_noise(text, config.ocr_probability, config.ocr_insert_space_prob, rng)
    elif scenario == DegradationScenario.MIXED:
        degraded = introduce_wrong_diacritics(
            text, config.wrong_diacritic_probability, config.wrong_swap_probability, rng
        )
        return apply_ocr_noise(degraded, config.ocr_probability, config.ocr_insert_space_prob, rng)
    else:
        return text


# Batch generation

@dataclass
class DegradedSample:
    input: str             # degraded text (model input)
    target: str            # clean text (model target / ground truth)
    scenario: str          # which degradation was applied
    changed_positions: list[int] = field(default_factory=list)  # char indices that changed


def generate_samples(
    texts: list[str],
    config: DegradationConfig,
    per_text_variants: int = 1,
) -> list[DegradedSample]:
    samples: list[DegradedSample] = []
    scenarios = [
        DegradationScenario.STRIP_ALL,
        DegradationScenario.STRIP_PARTIAL,
        DegradationScenario.WRONG_DIACRITICS,
        DegradationScenario.CEDILLA,
        DegradationScenario.OCR,
        DegradationScenario.MIXED,
    ]
    for i, text in enumerate(texts):
        for variant_idx, scenario in enumerate(scenarios):
            cfg = DegradationConfig(
                scenario=scenario,
                strip_probability=config.strip_probability,
                wrong_diacritic_probability=config.wrong_diacritic_probability,
                wrong_swap_probability=config.wrong_swap_probability,
                ocr_probability=config.ocr_probability,
                ocr_insert_space_prob=config.ocr_insert_space_prob,
                mixed_weights=config.mixed_weights,
                seed=(config.seed + i * 10 + variant_idx) if config.seed is not None else None,
            )
            degraded, used_scenario = degrade_text(text, cfg)
            changed = _find_changed_positions(text, degraded)
            samples.append(
                DegradedSample(
                    input=degraded,
                    target=text,
                    scenario=used_scenario.value,
                    changed_positions=changed,
                )
            )
    return samples


def _find_changed_positions(original: str, degraded: str) -> list[int]:
    """Return character indices where original and degraded differ (best-effort, same length)."""
    if len(original) != len(degraded):
        return []  # length changed (OCR merge/split), skip for now
    return [i for i, (a, b) in enumerate(zip(original, degraded)) if a != b]