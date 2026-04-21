"""
Fine-tune ByT5 for Romanian diacritics restoration.

Outputs in `models/byt5-diacritics/`:
    - checkpoint-*/            (HF checkpoints)
    - final/                   (best model, tokenizer, config)
    - training_log.csv         (per-step loss + periodic eval CER/WER)
    - eval_by_scenario.csv     (final per-scenario breakdown on full val set)
    - train_args.json          (the exact args this run was launched with)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

# Make `app` importable when running as `python -m scripts.train`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.dataset import DiacriticsDataset  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics (CER / WER) — no external dependency
# ---------------------------------------------------------------------------

def _edit_distance(a: list | str, b: list | str) -> int:
    """Levenshtein distance via two-row DP."""
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
    return _edit_distance(pred, target) / len(target)


def wer(pred: str, target: str) -> float:
    tgt_tokens = target.split()
    if not tgt_tokens:
        return 0.0 if not pred.split() else 1.0
    return _edit_distance(pred.split(), tgt_tokens) / len(tgt_tokens)


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id) 
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        cers = [cer(p.strip(), l.strip()) for p, l in zip(decoded_preds, decoded_labels)]
        wers = [wer(p.strip(), l.strip()) for p, l in zip(decoded_preds, decoded_labels)]
        return {"cer": float(np.mean(cers)), "wer": float(np.mean(wers))}
    return compute_metrics


# ---------------------------------------------------------------------------
# CSV logging callback (plain stdlib csv — no tensorboard, no wandb)
# ---------------------------------------------------------------------------

class CSVLoggerCallback(TrainerCallback):
    """Writes every Trainer log event as a CSV row.

    The file is rewritten on each flush because eval-time logs introduce new
    columns (eval_loss, eval_cer, ...) that don't exist in train-time logs.
    This keeps the column set a stable union without per-row JSON parsing.
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {"step": state.global_step, "epoch": state.epoch, **logs}
        self.rows.append(row)
        self._flush()

    def _flush(self):
        if not self.rows:
            return
        fieldnames: list[str] = []
        seen: set[str] = set()
        for r in self.rows:
            for k in r:
                if k not in seen:
                    fieldnames.append(k)
                    seen.add(k)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

    def on_train_end(self, args, state, control, **kwargs):
        self._flush()


# ---------------------------------------------------------------------------
# Post-training per-scenario evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_by_scenario(
    model,
    tokenizer,
    val_dataset: DiacriticsDataset,
    output_csv: Path,
    batch_size: int = 8,
    max_input_length: int = 1024,
    max_output_length: int = 1024,
    num_beams: int = 1,
    device: str = "cuda",
):
    """Run generation on the full val set and break down CER/WER by scenario."""
    from collections import defaultdict

    model.eval()
    triples = val_dataset.raw_triples()
    cers_by_scenario: dict[str, list[float]] = defaultdict(list)
    wers_by_scenario: dict[str, list[float]] = defaultdict(list)

    for i in range(0, len(triples), batch_size):
        batch = triples[i : i + batch_size]
        inputs_text = [t[0] for t in batch]
        targets_text = [t[1] for t in batch]
        scenarios = [t[2] for t in batch]

        enc = tokenizer(
            inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        out = model.generate(
            **enc,
            max_new_tokens=max_output_length,
            num_beams=num_beams,
        )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        for pred, tgt, sc in zip(decoded, targets_text, scenarios):
            cers_by_scenario[sc].append(cer(pred.strip(), tgt.strip()))
            wers_by_scenario[sc].append(wer(pred.strip(), tgt.strip()))

        if (i // batch_size) % 20 == 0:
            logger.info("  eval_by_scenario: %d / %d", i + len(batch), len(triples))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "n", "cer", "wer"])
        all_cers: list[float] = []
        all_wers: list[float] = []
        for sc in sorted(cers_by_scenario):
            w.writerow([
                sc,
                len(cers_by_scenario[sc]),
                float(np.mean(cers_by_scenario[sc])),
                float(np.mean(wers_by_scenario[sc])),
            ])
            all_cers.extend(cers_by_scenario[sc])
            all_wers.extend(wers_by_scenario[sc])
        w.writerow([
            "OVERALL",
            len(all_cers),
            float(np.mean(all_cers)) if all_cers else 0.0,
            float(np.mean(all_wers)) if all_wers else 0.0,
        ])
    logger.info("Wrote per-scenario eval to %s", output_csv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ByT5 for Romanian diacritics.")

    # Data / model
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/val.jsonl")
    parser.add_argument("--model_name", type=str, default="google/byt5-base")
    parser.add_argument("--output_dir", type=str, default="models/byt5-diacritics")

    # Optimization
    parser.add_argument("--num_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Sequence lengths
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Max bytes per input. ByT5 tokenizes at byte level; a 400-char "
                             "Romanian sentence with diacritics is ~450-500 bytes.")
    parser.add_argument("--max_target_length", type=int, default=1024)

    # Eval / save / log cadence
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_subset_size", type=int, default=500,
                        help="During-training eval uses only the first N val examples "
                             "(for speed). Final per-scenario eval uses the full set.")

    # Debug / perf knobs
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Cap on the full val set (None = all).")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable bf16 mixed precision (use fp32).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- model + tokenizer ------------------------------------------------
    logger.info("Loading tokenizer and model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False   # required with gradient checkpointing

    # ---- datasets ---------------------------------------------------------
    logger.info("Building datasets…")
    train_ds = DiacriticsDataset(
        args.train_file, tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        max_samples=args.max_train_samples,
    )
    full_val_ds = DiacriticsDataset(
        args.val_file, tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        max_samples=args.max_eval_samples,
    )
    # During-training eval: cap to eval_subset_size (first N examples) for speed.
    eval_subset_size = min(args.eval_subset_size, len(full_val_ds))
    train_eval_ds = DiacriticsDataset(
        args.val_file, tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        max_samples=eval_subset_size,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # ---- training args ----------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=1,  # greedy during eval for speed; beam search at inference
        bf16=(not args.no_bf16) and torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=["none"],      # we handle logging via CSVLoggerCallback
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    csv_callback = CSVLoggerCallback(output_dir / "training_log.csv")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=train_eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[csv_callback],
    )

    # ---- train ------------------------------------------------------------
    logger.info(
        "Starting training: %d train examples, %d eval examples (of %d full val)",
        len(train_ds), len(train_eval_ds), len(full_val_ds),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ---- save final model -------------------------------------------------
    final_dir = output_dir / "final"
    logger.info("Saving final (best) model to %s", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ---- per-scenario eval on FULL val set --------------------------------
    logger.info("Running per-scenario evaluation on the full val set…")
    evaluate_by_scenario(
        trainer.model,
        tokenizer,
        full_val_ds,
        output_csv=output_dir / "eval_by_scenario.csv",
        batch_size=args.per_device_eval_batch_size,
        max_input_length=args.max_input_length,
        max_output_length=args.max_target_length,
        num_beams=1,
        device=str(trainer.args.device),
    )

    logger.info("Done. All outputs under %s", output_dir)


if __name__ == "__main__":
    main()
