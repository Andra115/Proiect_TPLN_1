"""
Plot the training curve from training_log.csv.

Produces a two-panel figure:
  - Top: training loss (smooth line) + eval loss (markers connected by line)
  - Bottom: eval CER and eval WER (markers connected by line, %)

Both x-axes share the optimizer step count.

Usage:
    python -m scripts.plot_training \
        --log models/byt5-diacritics/training_log.csv \
        --output report_artifacts/training_curve.png

The training_log.csv produced by scripts/train.py interleaves two row types:
    - "train" rows: step, epoch, loss, grad_norm, learning_rate, (empty eval cols)
    - "eval" rows:  step, epoch, (empty train cols), eval_loss, eval_cer, eval_wer, ...

This script separates the two and plots them on shared x axes.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _parse_float(s: str) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_log(path: Path) -> tuple[list[dict], list[dict]]:
    """Read training_log.csv and split into (train_rows, eval_rows)."""
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = _parse_float(row.get("step", ""))
            if step is None:
                continue
            loss = _parse_float(row.get("loss", ""))
            eval_loss = _parse_float(row.get("eval_loss", ""))
            if loss is not None:
                train_rows.append({
                    "step": int(step),
                    "loss": loss,
                    "lr": _parse_float(row.get("learning_rate", "")),
                })
            if eval_loss is not None:
                eval_rows.append({
                    "step": int(step),
                    "eval_loss": eval_loss,
                    "eval_cer": _parse_float(row.get("eval_cer", "")),
                    "eval_wer": _parse_float(row.get("eval_wer", "")),
                })
    return train_rows, eval_rows


def plot(train_rows: list[dict], eval_rows: list[dict], output_path: Path) -> None:
    if not train_rows:
        raise RuntimeError("No training rows found in log file")

    train_steps = [r["step"] for r in train_rows]
    train_losses = [r["loss"] for r in train_rows]

    eval_steps = [r["step"] for r in eval_rows]
    eval_losses = [r["eval_loss"] for r in eval_rows]
    eval_cers = [r["eval_cer"] * 100 for r in eval_rows]
    eval_wers = [r["eval_wer"] * 100 for r in eval_rows]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "figure.dpi": 100,
    })

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"hspace": 0.15, "height_ratios": [1.0, 1.0]},
    )

    ax_top.plot(
        train_steps, train_losses,
        color="#1f6fb4", linewidth=1.2, alpha=0.85,
        label="Training loss",
    )
    if eval_rows:
        ax_top.plot(
            eval_steps, eval_losses,
            color="#c8511c", marker="o", markersize=5,
            linewidth=1.5, linestyle="-",
            label="Validation loss",
        )

    ax_top.set_yscale("log")
    ax_top.set_ylabel("Loss (log scale)")
    ax_top.set_title("ByT5-base — Romanian diacritics restoration")
    ax_top.legend(loc="upper right")

    if eval_rows:
        ax_bot.plot(
            eval_steps, eval_cers,
            color="#2e7d32", marker="o", markersize=5,
            linewidth=1.5,
            label="Validation CER",
        )
        ax_bot.plot(
            eval_steps, eval_wers,
            color="#c8a96e", marker="s", markersize=5,
            linewidth=1.5,
            label="Validation WER",
        )

        if len(eval_cers) > 0:
            final_step = eval_steps[-1]
            final_cer = eval_cers[-1]
            final_wer = eval_wers[-1]
            ax_bot.annotate(
                f"CER {final_cer:.2f}%",
                xy=(final_step, final_cer),
                xytext=(-50, 12), textcoords="offset points",
                fontsize=9, color="#2e7d32",
                arrowprops=dict(arrowstyle="-", color="#2e7d32", lw=0.5),
            )
            ax_bot.annotate(
                f"WER {final_wer:.2f}%",
                xy=(final_step, final_wer),
                xytext=(-50, 12), textcoords="offset points",
                fontsize=9, color="#a87f3e",
                arrowprops=dict(arrowstyle="-", color="#c8a96e", lw=0.5),
            )

    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("Training step")
    ax_bot.set_ylabel("Error rate (%, log scale)")
    ax_bot.legend(loc="upper right")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    logger.info("Wrote %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=str,
                        default="models/byt5-diacritics/training_log.csv",
                        help="Path to training_log.csv")
    parser.add_argument("--output", type=str,
                        default="report_artifacts/training_curve.png",
                        help="Where to write the PNG")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    log_path = Path(args.log)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows, eval_rows = load_log(log_path)
    logger.info("Loaded %d train rows, %d eval rows from %s",
                len(train_rows), len(eval_rows), log_path)

    plot(train_rows, eval_rows, output_path)


if __name__ == "__main__":
    main()
