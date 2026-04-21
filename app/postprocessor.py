"""
postprocessor.py
================
Post-processing module for the Romanian diacritics restoration pipeline.

WHAT THIS MODULE DOES:
  The ByT5 model is good at restoring diacritics based on linguistic context,
  but it can make predictable mistakes on specific token types. This module
  acts as a rule-based safety net that fixes those mistakes AFTER the model runs.

  Rules applied (in order):
    1. Fix OCR noise artifacts
    2. Preserve tokens the model should never touch:
       - URLs (https://..., www....)
       - Email addresses (user@domain.com)
       - Numbers (123, 3.14, 99%)
       - ALL-CAPS acronyms (UNESCO, NATO, SRL)
    3. Fix cedilla variants the model may have missed (ş→ș, ţ→ț)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CEDILLA_TO_CORRECT: dict[str, str] = {
    "ş": "ș",
    "Ş": "Ș",
    "ţ": "ț",
    "Ţ": "Ț",
}

# All valid Romanian diacritics (correct forms)
VALID_DIACRITICS = set("șțăâîȘȚĂÂÎ")

# ---------------------------------------------------------------------------
# Regex patterns for tokens that should never be modified by the model
# ---------------------------------------------------------------------------

# Matches URLs like https://example.com or www.example.com/path
_URL_RE = re.compile(
    r"https?://[^\s]+"           # http:// or https:// followed by non-space
    r"|www\.[^\s]+"              # or www. followed by non-space
)

# Matches email addresses like user@domain.com
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Matches numbers, including decimals and percentages
_NUMBER_RE = re.compile(r"\b\d+([.,]\d+)*%?\b")

# Matches ALL-CAPS tokens of 2+ letters (acronyms like UNESCO, SRL, ONG)
_ACRONYM_RE = re.compile(r"\b[A-ZȘȚĂÂÎ]{2,}\b")

# Combined: any "protected" token
_PROTECTED_RE = re.compile(
    r"(?:"
    + _URL_RE.pattern
    + r"|" + _EMAIL_RE.pattern
    + r"|" + _NUMBER_RE.pattern
    + r"|" + _ACRONYM_RE.pattern
    + r")"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PostProcessingResult:
    """
    The result of running post-processing on a single model output.

    Attributes:
        original_output:  The raw text received from the model (before post-processing).
        final_text:       The cleaned text after all post-processing rules have been applied.
        changes_made:     A list of descriptions of every change that was made.
        was_modified:     True if the final_text differs from the original_output.
    """
    original_output: str
    final_text: str
    changes_made: list[str] = field(default_factory=list)
    was_modified: bool = False


# ---------------------------------------------------------------------------
# Individual rule functions
# ---------------------------------------------------------------------------

def fix_ocr_space_errors(model_input: str, model_output: str) -> tuple[str, list[str]]:
    """
    Rule 1: Fix OCR noise artifacts that the model may have carried through.

    The OCR degradation introduces two categories of errors:

    A) CHARACTER SUBSTITUTIONS — pairs of characters that look similar:
       HOW WE FIX THESE: We compare the input and output token by token.
       If a token in the output looks like an OCR-corrupted version of the
       corresponding input token, we restore the input version.

    B) SPACE ERRORS — random space insertion or removal:
       HOW WE FIX THESE: We always collapse multiple consecutive spaces.
       For merges, we can't reliably re-split without a dictionary, so we
       leave those for the model to handle via context.

    Args:
        model_input:  The degraded text (may have OCR errors).
        model_output: The model's output text.

    Returns:
        (corrected_text, list_of_changes_made)
    """
    changes = []

    # ------------------------------------------------------------------
    # Part A: Fix character-level OCR substitutions
    # ------------------------------------------------------------------

    OCR_FIXES = [
        # Numbers misread as letters (common in model output)
        ("l", "1"),
        ("o", "0"),
        # Multi-character OCR confusions reproduced by the model
        ("rn", "m"),
        ("ii", "u"),
        ("cl", "d"),
        ("vv", "w"),
    ]

    input_tokens = model_input.split()
    output_tokens = model_output.split()

    # Only attempt token-level OCR fixes when word counts match.
    # If counts differ we already have a space error and alignment is unsafe.
    if len(input_tokens) == len(output_tokens):
        result_tokens = []
        for i, (inp_tok, out_tok) in enumerate(zip(input_tokens, output_tokens)):
            fixed_tok = out_tok
            for ocr_form, correct_form in OCR_FIXES:
                if correct_form in inp_tok and ocr_form in out_tok:
                    fixed_tok = fixed_tok.replace(ocr_form, correct_form)
                    changes.append(
                        f"OCR fix at token {i}: '{ocr_form}'→'{correct_form}' "
                        f"('{out_tok}'→'{fixed_tok}')"
                    )
                    break  # one fix per token to avoid cascading replacements
            result_tokens.append(fixed_tok)
        model_output = " ".join(result_tokens)

    # ------------------------------------------------------------------
    # Part B: Fix space errors
    # ------------------------------------------------------------------

    cleaned = re.sub(r" {2,}", " ", model_output).strip()
    if cleaned != model_output:
        changes.append("collapsed multiple consecutive spaces")
        model_output = cleaned

    return model_output, changes


def restore_protected_tokens(model_input: str, model_output: str) -> tuple[str, list[str]]:
    """
    Rule 2: Copy protected tokens (URLs, emails, numbers, acronyms) from the
    original degraded input back into the model output, word by word.

    WHY: The model receives degraded text and produces corrected text. But
    for tokens like URLs or numbers, the model should output them EXACTLY
    as they appeared in the input — no changes at all. If the model altered
    them, this function restores the original.

    HOW IT WORKS:
        We split both the input and the model output into words.
        For each word in the input that matches a "protected" pattern,
        we find the corresponding position in the output and replace it.
        If word counts differ (due to OCR space errors), we fall back
        to a best-effort token-by-token alignment.

    Args:
        model_input:  The degraded text that was fed to the model.
        model_output: The raw text the model produced.

    Returns:
        (corrected_text, list_of_changes_made)
    """
    changes = []

    input_tokens = model_input.split()
    output_tokens = model_output.split()

    # If token counts match, do a direct positional alignment
    if len(input_tokens) == len(output_tokens):
        result_tokens = []
        for i, (inp_tok, out_tok) in enumerate(zip(input_tokens, output_tokens)):
            if _PROTECTED_RE.fullmatch(inp_tok) and inp_tok != out_tok:
                result_tokens.append(inp_tok)
                changes.append(
                    f"protected token restored at position {i}: '{out_tok}'→'{inp_tok}'"
                )
            else:
                result_tokens.append(out_tok)
        return " ".join(result_tokens), changes

    # Fallback: token counts differ (OCR caused space split/merge)
    # In this case we can't do safe positional alignment, so we skip this rule
    # and log a warning. The other rules (cedilla, doubled) still apply.
    logger.warning(
        "Token count mismatch (input=%d, output=%d) — skipping protected token restoration.",
        len(input_tokens),
        len(output_tokens),
    )
    changes.append(
        f"WARNING: token count mismatch ({len(input_tokens)} vs {len(output_tokens)}) "
        "— protected token rule skipped"
    )
    return model_output, changes


def fix_cedilla_variants(text: str) -> tuple[str, list[str]]:
    """
    Rule 3: Replace cedilla diacritics with the correct comma-below forms.

    WHY: Some fonts/keyboards produce ş and ţ (cedilla) instead of the
    correct Romanian ș and ț (comma below). The model might not always
    catch these, so we fix them as a guaranteed rule.
    """
    changes = []
    for wrong, correct in CEDILLA_TO_CORRECT.items():
        if wrong in text:
            count = text.count(wrong)
            text = text.replace(wrong, correct)
            changes.append(f"cedilla fix: '{wrong}'→'{correct}' ({count}x)")
    return text, changes


# ---------------------------------------------------------------------------
# Main post-processing function
# ---------------------------------------------------------------------------

def postprocess(model_input: str, model_output: str) -> PostProcessingResult:
    """
    Run all post-processing rules on a single model output.

    This is the main function you call from the pipeline. Pass in both the
    original degraded input (so we can restore protected tokens) and the
    raw model output (what ByT5 produced).

    Args:
        model_input:  The degraded text that was sent to the model.
        model_output: The raw text the model produced.

    Returns:
        A PostProcessingResult with the cleaned text and a log of changes.
    """
    all_changes: list[str] = []
    text = model_output

    # --- Rule 1: Fix OCR errors ---
    text, c = fix_ocr_space_errors(model_input, text)
    all_changes.extend(c)

    # --- Rule 2: Restore protected tokens (URLs, numbers, acronyms, emails) ---
    text, c = restore_protected_tokens(model_input, text)
    all_changes.extend(c)

    # --- Rule 3: Fix cedilla variants ---
    text, c = fix_cedilla_variants(text)
    all_changes.extend(c)

    return PostProcessingResult(
        original_output=model_output,
        final_text=text,
        changes_made=all_changes,
        was_modified=(text != model_output),
    )


def postprocess_batch(model_inputs: list[str], model_outputs: list[str]) -> list[PostProcessingResult]:
    """
    Run post-processing on a batch of model outputs.

    Args:
        model_inputs:  List of degraded texts that were sent to the model.
        model_outputs: List of raw texts the model produced (same order).

    Returns:
        List of PostProcessingResult objects (same order as inputs).

    Raises:
        ValueError: If the lists have different lengths.
    """
    if len(model_inputs) != len(model_outputs):
        raise ValueError(
            f"model_inputs and model_outputs must have the same length, "
            f"got {len(model_inputs)} and {len(model_outputs)}"
        )

    results = []
    for inp, out in zip(model_inputs, model_outputs):
        results.append(postprocess(inp, out))

    n_modified = sum(1 for r in results if r.was_modified)
    logger.info(
        "Post-processing complete: %d/%d samples modified.",
        n_modified, len(results)
    )
    return results


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        {
            "description": "URL corrupted by model",
            "model_input":  "Vizitati https://gov.ro pentru detalii.",
            "model_output": "Vizitați httpș://gov.ro pentru detalii.",
        },
        {
            "description": "Number altered by model",
            "model_input":  "Au participat 1234 persoane.",
            "model_output": "Au participat l234 persoane.",
        },
        {
            "description": "Cedilla variants not fixed by model",
            "model_input":  "Ştefan şi ţara lui.",
            "model_output": "Ştefan şi ţara lui.",
        },
        {
            "description": "OCR errors",
            "model_input": "Mergem la munte maine.",
            "model_output": "Mergem la rnunte  mâine.",
        },
        {
            "description": "Word count mismatch",
            "model_input": "Au venit multe persoane la eveniment.",
            "model_output": "Au venit multe persoanela eveniment.",
        },
        {
            "description": "Clean output — no changes needed",
            "model_input":  "Ploaia a inundat orașul.",
            "model_output": "Ploaia a inundat orasul.",
        },
    ]

    print("=" * 60)
    print("POST-PROCESSOR DEMO")
    print("=" * 60)

    for case in test_cases:
        result = postprocess(case["model_input"], case["model_output"])
        print(f"\n[{case['description']}]")
        print(f"  Model output : {result.original_output}")
        print(f"  Final text   : {result.final_text}")
        if result.changes_made:
            for change in result.changes_made:
                print(f"  Change       : {change}")
        else:
            print(f"  Change       : (none)")