"""
Standalone script to generate a degraded dataset and save it to disk.
Useful for pre-generating training data without running the API.

Usage:
    python scripts/generate_dataset.py \
        --source demo \
        --output data/train.jsonl \
        --variants 3 \
        --scenario mixed \
        --seed 42

For real data:
    python scripts/generate_dataset.py \
        --source rolargesum \
        --max-sentences 50000 \
        --output data/train.jsonl \
        --variants 2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.degradation import DegradationConfig, DegradationScenario, generate_samples
from app.data_loader import DEMO_SENTENCES, load_rolargesum, load_text_file


def main():
    parser = argparse.ArgumentParser(description="Generate degraded Romanian text dataset.")
    parser.add_argument("--source", default="demo",
                        choices=["demo", "rolargesum", "file"],
                        help="Data source")
    parser.add_argument("--file", default=None, help="Path to text file (when --source file)")
    parser.add_argument("--max-sentences", type=int, default=10_000)
    parser.add_argument("--output", default="data/train.jsonl")
    parser.add_argument("--variants", type=int, default=1,
                        help="Degraded variants per input sentence")
    parser.add_argument("--scenario", default="mixed",
                        choices=[s.value for s in DegradationScenario])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strip-prob", type=float, default=0.5)
    parser.add_argument("--wrong-diac-prob", type=float, default=0.3)
    parser.add_argument("--wrong-swap-prob", type=float, default=0.2)
    parser.add_argument("--ocr-prob", type=float, default=0.02)
    args = parser.parse_args()

    # Load texts
    print(f"[1/3] Loading data from source: {args.source}")
    if args.source == "demo":
        texts = DEMO_SENTENCES
    elif args.source == "rolargesum":
        texts = load_rolargesum(max_sentences=args.max_sentences)
    elif args.source == "file":
        if not args.file:
            print("ERROR: --file required when --source file")
            sys.exit(1)
        texts = load_text_file(args.file, max_sentences=args.max_sentences)
    else:
        texts = DEMO_SENTENCES

    print(f"    → {len(texts)} sentences loaded")

    # Configure degradation
    config = DegradationConfig(
        scenario=DegradationScenario(args.scenario),
        strip_probability=args.strip_prob,
        wrong_diacritic_probability=args.wrong_diac_prob,
        wrong_swap_probability=args.wrong_swap_prob,
        ocr_probability=args.ocr_prob,
        seed=args.seed,
    )

    # Generate
    print(f"[2/3] Generating degraded samples ({args.variants} variant(s) per sentence)…")
    samples = generate_samples(texts, config, per_text_variants=args.variants)
    print(f"    → {len(samples)} total samples")

    # Write JSONL
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[3/3] Writing to {out_path}…")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({
                "input": s.input,
                "target": s.target,
                "scenario": s.scenario,
                "changed_positions": s.changed_positions,
            }, ensure_ascii=False) + "\n")

    print(f"Done! {len(samples)} samples → {out_path}")

    # Print a few examples
    print("\n--- Sample preview (first 3) ---")
    for s in samples[:3]:
        print(f"  scenario : {s.scenario}")
        print(f"  target   : {s.target}")
        print(f"  input    : {s.input}")
        print(f"  changes  : {len(s.changed_positions)} chars changed")
        print()


if __name__ == "__main__":
    main()