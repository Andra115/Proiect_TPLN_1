"""
detection/diff.py

Character-level alignment between degraded input and corrected output.
Uses SequenceMatcher so it handles insertions/deletions gracefully
(relevant for OCR split/merge errors from scenario 3.1).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class Change:
    position: int        # char index in the corrected output
    original_char: str   # what was in the degraded input
    corrected_char: str  # what the model (or mock) produced
    context: str         # ±15 chars of surrounding text from output
    change_type: str     # 'diacritic_added' | 'diacritic_swapped' | 'other'


# ---------------------------------------------------------------------------
# Known diacritic confusion pairs  (input_char → corrected_char)
# ---------------------------------------------------------------------------
_DIACRITIC_ADDITIONS: dict[tuple[str, str], str] = {
    # stripping confusions: base → accented
    ("s", "ș"): "diacritic_added",
    ("t", "ț"): "diacritic_added",
    ("a", "ă"): "diacritic_added",
    ("a", "â"): "diacritic_added",
    ("i", "î"): "diacritic_added",
    # cedilla → comma-below (legacy encoding fix)
    ("ş", "ș"): "diacritic_swapped",
    ("ţ", "ț"): "diacritic_swapped",
    # wrong diacritic substitutions
    ("ș", "s"): "diacritic_added",   # over-corrected input
    ("ț", "t"): "diacritic_added",
    ("ă", "a"): "diacritic_added",
    ("î", "i"): "diacritic_added",
    ("â", "a"): "diacritic_added",
}


def _classify_change(orig: str, corr: str) -> str:
    key = (orig.lower(), corr.lower())
    return _DIACRITIC_ADDITIONS.get(key, "other")


def char_level_diff(input_text: str, output_text: str) -> list[Change]:
    """
    Align input_text and output_text at character level and return every
    position where a character was substituted.

    Insertions / deletions (OCR merge/split artefacts) are recorded as
    change_type='other' with original_char='' or corrected_char='' so
    downstream scorers can handle them separately.
    """
    matcher = difflib.SequenceMatcher(None, input_text, output_text, autojunk=False)
    changes: list[Change] = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue

        if op == "replace":
            in_seg = input_text[i1:i2]
            out_seg = output_text[j1:j2]
            # Pair characters up; leftovers treated as insert/delete
            for k, (orig, corr) in enumerate(zip(in_seg, out_seg)):
                if orig != corr:
                    pos = j1 + k
                    context = output_text[max(0, pos - 15) : pos + 15]
                    changes.append(
                        Change(
                            position=pos,
                            original_char=orig,
                            corrected_char=corr,
                            context=context,
                            change_type=_classify_change(orig, corr),
                        )
                    )
            # Handle length mismatch (insert or delete within a replace block)
            len_diff = len(out_seg) - len(in_seg)
            if len_diff > 0:
                for k in range(len(in_seg), len(out_seg)):
                    pos = j1 + k
                    context = output_text[max(0, pos - 15) : pos + 15]
                    changes.append(
                        Change(pos, "", out_seg[k], context, "other")
                    )
            elif len_diff < 0:
                for k in range(len(out_seg), len(in_seg)):
                    pos = j1 + min(k, len(out_seg) - 1)
                    context = output_text[max(0, pos - 15) : pos + 15]
                    changes.append(
                        Change(pos, in_seg[k], "", context, "other")
                    )

        elif op == "insert":
            for k, corr in enumerate(output_text[j1:j2]):
                pos = j1 + k
                context = output_text[max(0, pos - 15) : pos + 15]
                changes.append(Change(pos, "", corr, context, "other"))

        elif op == "delete":
            # Deletion: character existed in input but not in output
            pos = j1  # best anchor we have
            for orig in input_text[i1:i2]:
                context = output_text[max(0, pos - 15) : pos + 15]
                changes.append(Change(pos, orig, "", context, "other"))

    return changes