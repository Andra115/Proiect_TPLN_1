"""
Evaluate a trained ByT5 diacritics-restoration model on a JSONL dataset.

Reads examples with the schema produced by scripts/generate_dataset.py:
    {"input": <degraded>, "target": <clean>, "scenario": <str>, "changed_positions": [int, ...]}

Produces (in --output directory):

    eval_overall.csv         Overall CER/WER + per-scenario breakdown
    eval_by_confusion.csv    Per confusion pair (ș/s, ț/t, ă/a, î/i, â/a, …) accuracy
    eval_detection.csv       Detection precision / recall / F1, per scenario
    eval_worst_errors.jsonl  Top-N predictions sorted by CER (for error analysis)
    eval_predictions.jsonl   Raw model predictions (cache — reuse with --use-cache)
    eval_summary.md          One-page markdown summary

Usage:
    python -m scripts.evaluate \\
        --model models/byt5-diacritics/final \\
        --data data/val.jsonl \\
        --output report_artifacts/eval \\
        --max-samples 3000

Subsequent re-runs of metrics-only (skip inference):
    python -m scripts.evaluate \\
        --model models/byt5-diacritics/final \\
        --data data/val.jsonl \\
        --output report_artifacts/eval \\
        --use-cache
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


CONFUSION_PAIRS: list[tuple[str, set[str]]] = [
    ("ș", {"s", "ş"}),
    ("ț", {"t", "ţ"}),
    ("ă", {"a"}),     
    ("â", {"a", "î"}),
    ("î", {"i", "â"}),
    ("Ș", {"S", "Ş"}),
    ("Ț", {"T", "Ţ"}),
    ("Ă", {"A"}),
    ("Â", {"A", "Î"}),
    ("Î", {"I", "Â"}),
]

# Flat set of "diacritic-relevant" characters for quick filtering
_DIACRITIC_CHARS = {c for c, _ in CONFUSION_PAIRS}
_DEGRADED_CHARS = {d for _, ds in CONFUSION_PAIRS for d in ds}
_ALL_RELEVANT = _DIACRITIC_CHARS | _DEGRADED_CHARS


def edit_distance(a, b) -> int:
    """Levenshtein distance via two-row DP. Works on strings or token lists."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def cer(pred: str, target: str) -> float:
    if not target:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, target) / len(target)


def wer(pred: str, target: str) -> float:
    tgt_tokens = target.split()
    if not tgt_tokens:
        return 0.0 if not pred.split() else 1.0
    return edit_distance(pred.split(), tgt_tokens) / len(tgt_tokens)


def generate_predictions(
    model,
    tokenizer,
    examples: list[dict],
    batch_size: int,
    max_input_length: int,
    max_output_length: int,
    device: str,
) -> list[str]:
    """Generate predictions for all examples, in batches, with progress logging."""
    predictions: list[str] = []
    n = len(examples)
    t_start = time.perf_counter()

    model.eval()
    for batch_start in range(0, n, batch_size):
        batch = examples[batch_start : batch_start + batch_size]
        inputs = [ex["input"] for ex in batch]

        enc = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_output_length,
                num_beams=1,
                do_sample=False,
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        predictions.extend(decoded)

        # Log progress every ~20 batches
        if (batch_start // batch_size) % 20 == 0 or batch_start + batch_size >= n:
            done = min(batch_start + batch_size, n)
            elapsed = time.perf_counter() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else 0
            logger.info(
                "  predicted %d / %d  (%.1f samples/s, ETA %.0fs)",
                done, n, rate, eta,
            )

    return predictions


def compute_overall(rows: list[dict]) -> dict[str, dict]:
    """Returns {scenario: {n, cer, wer}} plus an OVERALL entry."""
    by_scenario: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        c = cer(row["prediction"].strip(), row["target"].strip())
        w = wer(row["prediction"].strip(), row["target"].strip())
        by_scenario[row.get("scenario", "unknown")].append((c, w))

    out: dict[str, dict] = {}
    all_c: list[float] = []
    all_w: list[float] = []
    for sc, pairs in sorted(by_scenario.items()):
        cs = [c for c, _ in pairs]
        ws = [w for _, w in pairs]
        out[sc] = {
            "n": len(pairs),
            "cer": float(np.mean(cs)),
            "wer": float(np.mean(ws)),
        }
        all_c.extend(cs)
        all_w.extend(ws)
    out["OVERALL"] = {
        "n": len(rows),
        "cer": float(np.mean(all_c)) if all_c else 0.0,
        "wer": float(np.mean(all_w)) if all_w else 0.0,
    }
    return out


def compute_per_confusion(rows: list[dict]) -> dict[str, dict]:
    """For each (correct, degraded) pair, count how often:
       - the position was actually degraded in the input
       - the model restored it correctly

    Only counts positions in examples where len(input) == len(target) == len(prediction)
    (the strip/wrong-diacritics/cedilla scenarios, which is most of the data).
    """
    # counters[(correct_char, degraded_char)] -> {degraded_count, restored_count}
    counters: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"degraded_count": 0, "restored_count": 0}
    )
    skipped = 0

    pair_lookup: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for correct, degraded_set in CONFUSION_PAIRS:
        for d in degraded_set:
            pair_lookup[correct].append((correct, d))

    for row in rows:
        inp, tgt, pred = row["input"], row["target"], row["prediction"]
        if not (len(inp) == len(tgt) == len(pred)):
            skipped += 1
            continue

        for i, tgt_char in enumerate(tgt):
            if tgt_char not in _DIACRITIC_CHARS:
                continue
            inp_char = inp[i]
            # Was this position actually degraded?
            possible_pairs = pair_lookup.get(tgt_char, [])
            matched_pair = next(
                ((c, d) for c, d in possible_pairs if d == inp_char),
                None,
            )
            if matched_pair is None:
                continue  # input already had the correct character at this position
            counters[matched_pair]["degraded_count"] += 1
            if pred[i] == tgt_char:
                counters[matched_pair]["restored_count"] += 1

    out: dict[str, dict] = {}
    for (correct, degraded), c in sorted(counters.items()):
        total = c["degraded_count"]
        restored = c["restored_count"]
        acc = restored / total if total > 0 else 0.0
        key = f"{degraded}->{correct}"
        out[key] = {
            "correct_char": correct,
            "degraded_char": degraded,
            "n_degraded": total,
            "n_restored": restored,
            "accuracy": acc,
            "error_rate": 1.0 - acc,
        }
    out["_skipped_length_mismatch"] = skipped
    return out


def compute_detection_f1(rows: list[dict]) -> dict[str, dict]:
    """Detection metrics: how well the model identifies which positions are wrong.

    Gold edits  = positions in input where input[i] != target[i]
    Pred edits  = positions where input[i] != prediction[i]

    Only counts examples where len(input) == len(prediction) (so positions align).
    Examples where lengths differ are skipped (typically OCR with space changes).
    """
    by_scenario: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "n_examples": 0, "n_skipped": 0}
    )

    for row in rows:
        inp, tgt, pred = row["input"], row["target"], row["prediction"]
        sc = row.get("scenario", "unknown")
        bucket = by_scenario[sc]

        if not (len(inp) == len(pred) == len(tgt)):
            bucket["n_skipped"] += 1
            continue

        gold_edits = {i for i, (a, b) in enumerate(zip(inp, tgt)) if a != b}
        pred_edits = {i for i, (a, b) in enumerate(zip(inp, pred)) if a != b}

        bucket["tp"] += len(gold_edits & pred_edits)
        bucket["fp"] += len(pred_edits - gold_edits)
        bucket["fn"] += len(gold_edits - pred_edits)
        bucket["n_examples"] += 1

    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    out: dict[str, dict] = {}
    agg_tp = agg_fp = agg_fn = agg_n = agg_skipped = 0
    for sc, b in sorted(by_scenario.items()):
        p, r, f1 = _prf(b["tp"], b["fp"], b["fn"])
        out[sc] = {
            "n_examples": b["n_examples"],
            "n_skipped": b["n_skipped"],
            "tp": b["tp"], "fp": b["fp"], "fn": b["fn"],
            "precision": p, "recall": r, "f1": f1,
        }
        agg_tp += b["tp"]; agg_fp += b["fp"]; agg_fn += b["fn"]
        agg_n += b["n_examples"]; agg_skipped += b["n_skipped"]

    p, r, f1 = _prf(agg_tp, agg_fp, agg_fn)
    out["OVERALL"] = {
        "n_examples": agg_n, "n_skipped": agg_skipped,
        "tp": agg_tp, "fp": agg_fp, "fn": agg_fn,
        "precision": p, "recall": r, "f1": f1,
    }
    return out


def find_worst_errors(rows: list[dict], n: int = 20) -> list[dict]:
    """Return the n examples with highest CER, sorted descending."""
    scored = []
    for row in rows:
        c = cer(row["prediction"].strip(), row["target"].strip())
        if c > 0:
            scored.append((c, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for c, row in scored[:n]:
        out.append({
            "scenario": row.get("scenario", "unknown"),
            "cer": c,
            "wer": wer(row["prediction"].strip(), row["target"].strip()),
            "input": row["input"],
            "target": row["target"],
            "prediction": row["prediction"],
        })
    return out




def write_overall_csv(metrics: dict[str, dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "n", "cer", "wer"])
        for sc, m in metrics.items():
            w.writerow([sc, m["n"], f"{m['cer']:.6f}", f"{m['wer']:.6f}"])


def write_confusion_csv(metrics: dict[str, dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "degraded->correct", "correct_char", "degraded_char",
            "n_degraded", "n_restored", "accuracy", "error_rate",
        ])
        for key, m in metrics.items():
            if key.startswith("_"):
                continue
            w.writerow([
                key, m["correct_char"], m["degraded_char"],
                m["n_degraded"], m["n_restored"],
                f"{m['accuracy']:.6f}", f"{m['error_rate']:.6f}",
            ])


def write_detection_csv(metrics: dict[str, dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "n_examples", "n_skipped",
            "tp", "fp", "fn", "precision", "recall", "f1",
        ])
        for sc, m in metrics.items():
            w.writerow([
                sc, m["n_examples"], m["n_skipped"],
                m["tp"], m["fp"], m["fn"],
                f"{m['precision']:.6f}", f"{m['recall']:.6f}", f"{m['f1']:.6f}",
            ])


def write_errors_jsonl(errors: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def write_predictions_cache(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_predictions_cache(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_summary_md(
    args,
    overall: dict[str, dict],
    confusion: dict[str, dict],
    detection: dict[str, dict],
    worst: list[dict],
    path: Path,
) -> None:
    """Markdown summary"""
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"- **Model:** `{args.model}`")
    lines.append(f"- **Dataset:** `{args.data}` ({overall['OVERALL']['n']} examples)")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Overall and per-scenario CER / WER")
    lines.append("")
    lines.append("| Scenario | n | CER | WER |")
    lines.append("|---|---:|---:|---:|")
    for sc, m in overall.items():
        lines.append(
            f"| `{sc}` | {m['n']} | {m['cer'] * 100:.2f}% | {m['wer'] * 100:.2f}% |"
        )
    lines.append("")

    lines.append("## Per-confusion restoration accuracy")
    lines.append("")
    lines.append("How often the model produced the correct diacritic, given the input")
    lines.append("had a degraded form. Counts only length-preserving examples")
    lines.append(f"({confusion.get('_skipped_length_mismatch', 0)} skipped due to length mismatch).")
    lines.append("")
    lines.append("| Confusion | Times degraded | Restored | Accuracy | Error rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, m in confusion.items():
        if key.startswith("_"):
            continue
        lines.append(
            f"| `{key}` | {m['n_degraded']} | {m['n_restored']} | "
            f"{m['accuracy'] * 100:.2f}% | {m['error_rate'] * 100:.2f}% |"
        )
    lines.append("")

    lines.append("## Detection precision / recall / F1")
    lines.append("")
    lines.append("Treats each character position as a binary classification:")
    lines.append("did the model correctly identify whether this position needed correction?")
    lines.append("")
    lines.append("| Scenario | n | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for sc, m in detection.items():
        lines.append(
            f"| `{sc}` | {m['n_examples']} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{m['precision'] * 100:.2f}% | {m['recall'] * 100:.2f}% | "
            f"{m['f1'] * 100:.2f}% |"
        )
    lines.append("")

    lines.append("## Top error examples")
    lines.append("")
    lines.append("The worst predictions by CER, useful for qualitative analysis.")
    lines.append("Full list in `eval_worst_errors.jsonl`.")
    lines.append("")
    for i, e in enumerate(worst[:10], 1):
        lines.append(f"### {i}. `{e['scenario']}` — CER {e['cer'] * 100:.2f}%")
        lines.append("")
        lines.append(f"- **Input:**     {e['input']}")
        lines.append(f"- **Target:**    {e['target']}")
        lines.append(f"- **Predicted:** {e['prediction']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")



def print_summary(overall: dict, detection: dict) -> None:
    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall:  CER {overall['OVERALL']['cer'] * 100:.2f}%   "
          f"WER {overall['OVERALL']['wer'] * 100:.2f}%   "
          f"(n={overall['OVERALL']['n']})")
    print(f"Detection: P {detection['OVERALL']['precision'] * 100:.2f}%   "
          f"R {detection['OVERALL']['recall'] * 100:.2f}%   "
          f"F1 {detection['OVERALL']['f1'] * 100:.2f}%")
    print()
    print("Per-scenario CER:")
    for sc, m in overall.items():
        if sc == "OVERALL":
            continue
        print(f"  {sc:20s}  CER {m['cer'] * 100:6.2f}%   WER {m['wer'] * 100:6.2f}%   n={m['n']}")
    print()



def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, required=True,
                   help="Path to the trained model directory (e.g. models/byt5-diacritics/final)")
    p.add_argument("--data", type=str, required=True,
                   help="Path to the JSONL eval data file (val.jsonl or gold.jsonl)")
    p.add_argument("--output", type=str, default="report_artifacts/eval",
                   help="Output directory for the eval CSVs / JSONL / markdown")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit evaluation to the first N examples (default: full file)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Generation batch size (default: 8)")
    p.add_argument("--max-input-length", type=int, default=512)
    p.add_argument("--max-output-length", type=int, default=512)
    p.add_argument("--top-errors", type=int, default=20,
                   help="How many worst-CER examples to record (default: 20)")
    p.add_argument("--use-cache", action="store_true",
                   help="Skip inference and load predictions from eval_predictions.jsonl "
                        "in the output dir. Useful for iterating on metrics.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load eval examples
    data_path = Path(args.data)
    examples: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if args.max_samples is not None and len(examples) >= args.max_samples:
                break
    logger.info("Loaded %d examples from %s", len(examples), data_path)

    cache_path = output_dir / "eval_predictions.jsonl"

    if args.use_cache and cache_path.exists():
        logger.info("Loading cached predictions from %s", cache_path)
        rows = load_predictions_cache(cache_path)
        if len(rows) != len(examples):
            logger.warning(
                "Cache has %d rows but data has %d examples — using the cache as-is.",
                len(rows), len(examples),
            )
    else:
        # Load model
        logger.info("Loading model from %s ...", args.model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
        logger.info("Model on %s.", device)

        logger.info("Generating predictions on %d examples ...", len(examples))
        predictions = generate_predictions(
            model, tokenizer, examples,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            device=device,
        )

        rows = [{**ex, "prediction": pred} for ex, pred in zip(examples, predictions)]
        write_predictions_cache(rows, cache_path)
        logger.info("Wrote predictions cache to %s", cache_path)

        # Free GPU memory before metric computation
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute metrics
    logger.info("Computing overall + per-scenario CER/WER ...")
    overall = compute_overall(rows)

    logger.info("Computing per-confusion accuracy ...")
    confusion = compute_per_confusion(rows)

    logger.info("Computing detection precision/recall/F1 ...")
    detection = compute_detection_f1(rows)

    logger.info("Finding top %d worst errors ...", args.top_errors)
    worst = find_worst_errors(rows, n=args.top_errors)

    # Write outputs
    write_overall_csv(overall,     output_dir / "eval_overall.csv")
    write_confusion_csv(confusion, output_dir / "eval_by_confusion.csv")
    write_detection_csv(detection, output_dir / "eval_detection.csv")
    write_errors_jsonl(worst,      output_dir / "eval_worst_errors.jsonl")
    write_summary_md(args, overall, confusion, detection, worst,
                     output_dir / "eval_summary.md")

    logger.info("All outputs written under %s/", output_dir)
    print_summary(overall, detection)


if __name__ == "__main__":
    main()