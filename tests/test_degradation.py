"""
Tests for the degradation pipeline.
Run with: pytest tests/test_degradation.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import pytest
from app.degradation import (
    DegradationConfig,
    DegradationScenario,
    degrade_text,
    generate_samples,
    strip_all_diacritics,
    strip_partial_diacritics,
    apply_cedilla_variants,
    introduce_wrong_diacritics,
    DIACRITIC_TO_BASE,
    CEDILLA_TO_CORRECT,
)
import random


CLEAN_SENTENCES = [
    "Ștefan cel Mare a fost un domnitor important.",
    "Câmpul de grâu se întindea până la munte.",
    "Ploaia torențială a inundat străzile orașului.",
    "Îngrijitorii hrănesc animalele în fiecare zi.",
    "Fântâna din centrul parcului a fost renovată.",
]


# ---------------------------------------------------------------------------
# strip_all_diacritics
# ---------------------------------------------------------------------------

def test_strip_all_removes_all_diacritics():
    text = "Ștefan câmpul grâu întindea până"
    result = strip_all_diacritics(text)
    for diac in DIACRITIC_TO_BASE:
        assert diac not in result, f"Diacritic {diac!r} still present after strip_all"


def test_strip_all_preserves_non_diacritic_chars():
    text = "Hello world 123 !@#"
    assert strip_all_diacritics(text) == text


def test_strip_all_case_sensitive():
    assert strip_all_diacritics("Ș") == "S"
    assert strip_all_diacritics("ș") == "s"
    assert strip_all_diacritics("Ț") == "T"
    assert strip_all_diacritics("ț") == "t"


# ---------------------------------------------------------------------------
# strip_partial_diacritics
# ---------------------------------------------------------------------------

def test_strip_partial_always_removes_at_prob_1():
    rng = random.Random(0)
    text = "Ștefan câmpul grâu"
    result = strip_partial_diacritics(text, probability=1.0, rng=rng)
    for diac in DIACRITIC_TO_BASE:
        assert diac not in result


def test_strip_partial_never_removes_at_prob_0():
    rng = random.Random(0)
    text = "Ștefan câmpul grâu"
    result = strip_partial_diacritics(text, probability=0.0, rng=rng)
    assert result == text


def test_strip_partial_reproducible_with_seed():
    text = "Ploaia torențială a inundat."
    r1 = strip_partial_diacritics(text, 0.5, random.Random(42))
    r2 = strip_partial_diacritics(text, 0.5, random.Random(42))
    assert r1 == r2


# ---------------------------------------------------------------------------
# cedilla
# ---------------------------------------------------------------------------

def test_cedilla_replaces_correct_with_cedilla():
    rng = random.Random(0)
    text = "Ștefan și Țara"
    result = apply_cedilla_variants(text, rng, probability=1.0)
    # ș→ş, ț→ţ
    assert "ş" in result or "ţ" in result
    assert "ș" not in result
    assert "ț" not in result


def test_cedilla_does_not_touch_base_chars():
    rng = random.Random(0)
    text = "Stefan si Tara"  # no diacritics at all
    result = apply_cedilla_variants(text, rng, probability=1.0)
    assert result == text


# ---------------------------------------------------------------------------
# introduce_wrong_diacritics
# ---------------------------------------------------------------------------

def test_wrong_diacritics_output_has_some_diacritics():
    rng = random.Random(0)
    text = "Stefan campul grau"  # all base, no diacritics
    result = introduce_wrong_diacritics(text, wrong_diacritic_prob=1.0, wrong_swap_prob=0.0, rng=rng)
    # with prob=1.0, every eligible base char gets a diacritic
    diacritics_present = any(ch in result for ch in DIACRITIC_TO_BASE)
    assert diacritics_present


# ---------------------------------------------------------------------------
# degrade_text / generate_samples
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", list(DegradationScenario))
def test_degrade_returns_string(scenario):
    config = DegradationConfig(scenario=scenario, seed=0)
    degraded, used_scenario = degrade_text("Ștefan cel Mare a câștigat.", config)
    assert isinstance(degraded, str)
    assert len(degraded) > 0


def test_generate_samples_count():
    config = DegradationConfig(scenario=DegradationScenario.STRIP_ALL, seed=0)
    samples = generate_samples(CLEAN_SENTENCES, config, per_text_variants=3)
    assert len(samples) == len(CLEAN_SENTENCES) * 3


def test_generate_samples_target_unchanged():
    config = DegradationConfig(scenario=DegradationScenario.STRIP_ALL, seed=0)
    samples = generate_samples(CLEAN_SENTENCES, config)
    for original, sample in zip(CLEAN_SENTENCES, samples):
        assert sample.target == original


def test_strip_all_sample_input_has_no_diacritics():
    config = DegradationConfig(scenario=DegradationScenario.STRIP_ALL, seed=0)
    samples = generate_samples(CLEAN_SENTENCES, config)
    for s in samples:
        for diac in DIACRITIC_TO_BASE:
            assert diac not in s.input, f"Diacritic {diac!r} in input after strip_all"


def test_changed_positions_valid():
    config = DegradationConfig(scenario=DegradationScenario.STRIP_ALL, seed=0)
    samples = generate_samples(CLEAN_SENTENCES, config)
    for s in samples:
        # changed_positions should only exist when lengths match
        if s.changed_positions:
            assert len(s.input) == len(s.target)
            for pos in s.changed_positions:
                assert s.input[pos] != s.target[pos]


def test_mixed_scenario_uses_different_scenarios():
    config = DegradationConfig(scenario=DegradationScenario.MIXED, seed=None)
    texts = CLEAN_SENTENCES * 20
    samples = generate_samples(texts, config, per_text_variants=1)
    used = {s.scenario for s in samples}
    assert len(used) > 1, "MIXED should produce variety of scenarios"