# Romanian Diacritics Restoration

End-to-end system for restoring missing or incorrect diacritics in Romanian text,
built around a fine-tuned [ByT5-base](https://huggingface.co/google/byt5-base) model.
The system handles full and partial diacritic stripping, wrong-diacritic substitution,
legacy cedilla encoding (`ş`/`ţ` → `ș`/`ț`), and OCR-style noise. Each restored
character also comes with a confidence score derived from the model's own softmax
distribution, surfaced visually in the demo so the user can see which corrections
the model was less sure about.

Built for the *Tehnici de Prelucrare a Limbajului Natural* course at FII UAIC.

## Results

In-distribution (RoLargeSum-derived validation set, n=3,000) and out-of-distribution
(hand-verified Romanian Wikipedia gold set, n=360):

| Metric | Validation | Gold (OOD) |
|---|---:|---:|
| CER | **0.32%** | **0.98%** |
| WER | 1.70% | 5.16% |
| Detection F1 | 98.25% | 92.87% |

See [the project report](report.pdf) for per-scenario breakdowns, per-confusion accuracy,
and error analysis.

## Quick start

```bash
# Clone and set up
git clone https://github.com/Andra115/Proiect_TPLN_1.git
cd Proiect_TPLN_1
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The trained model isn't bundled with this repo (too large for git). Before
starting the demo, get the model in one of three ways:

```bash
# Option 1 - download the pre-trained checkpoint from HuggingFace into the
# default location the app looks for
pip install huggingface_hub
huggingface-cli download george-ge2/byt5-ro-diacritics \
    --local-dir models/byt5-diacritics/final

# Option 2 - point the app at the HF Hub repo directly (transformers will
# fetch and cache it on first run)
export DIACRITICS_MODEL_PATH=george-ge2/byt5-ro-diacritics

# Option 3 - train the model yourself; see "Reproducing the training run" below
```

Then start the demo:

```bash
cd app
uvicorn main:app --reload
# Open http://localhost:8000 in a browser
```

If the model isn't reachable when the server starts, the `/restore` endpoint
will return 503; everything else (data generation, post-processor) still works.

## Using the model directly

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("george-ge2/byt5-ro-diacritics")
model = AutoModelForSeq2SeqLM.from_pretrained("george-ge2/byt5-ro-diacritics")

text = "Comisia pentru Transport si Turism a votat luni favorabil propunerea ca incepand din anul 2021 sa se renunte la schimbarea orei de vara si de iarna in Uniunea Europeana."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512, num_beams=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# → "Comisia pentru Transport și Turism a votat luni favorabil propunerea ca începând din anul 2021 să se renunțe la schimbarea orei de vară și de iarnă în Uniunea Europeană."
```

For per-character confidence scores, use the wrapper class in
[`app/predict.py`](app/predict.py) instead of calling `generate` directly — it
handles greedy decoding with per-step softmax extraction and byte-to-character
probability aggregation.

## Project structure

```
app/
  main.py              FastAPI service + /restore endpoint
  predict.py           DiacriticsRestorer wrapper around ByT5
  dataset.py           PyTorch JSONL dataset for training
  degradation.py       Six-scenario degradation pipeline (data generator)
  postprocessor.py     Rule-based post-processor (URL/number protection, cedilla)
  static/index.html    Demo frontend with per-character confidence highlighting
scripts/
  train.py             HF Seq2SeqTrainer wrapper
  evaluate.py          CER/WER + per-confusion + detection F1 evaluation
  build_gold.py        Wikipedia-sourced gold set builder
  plot_training.py     Training curve plotter
data/
  train.jsonl          Generated training set (288k examples)
  val.jsonl            Held-out validation set
  gold.jsonl           Hand-verified OOD test set (360 examples)
models/
  byt5-diacritics/     Trained checkpoint (gitignored; on HF Hub)
```

## Reproducing the training run

```bash
# 1. Train (16 GB VRAM recommended)
python -m scripts.train \
    --train_file data/train.jsonl \
    --val_file data/val.jsonl \
    --output_dir models/byt5-diacritics \
    --max_train_samples 40000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-4 \
    --max_input_length 512 \
    --bf16

# 2. Evaluate on validation set
python -m scripts.evaluate \
    --model models/byt5-diacritics/final \
    --data data/val.jsonl \
    --output report_artifacts/eval

# 3. Build and evaluate on out-of-distribution gold set
python -m scripts.build_gold --num-sentences 60 --sleep 5.0
# (manually review data/gold_candidates.jsonl, then rename to data/gold.jsonl)
python -m scripts.evaluate \
    --model models/byt5-diacritics/final \
    --data data/gold.jsonl \
    --output report_artifacts/gold_eval
```

## API endpoints

The FastAPI service exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`  | `/` | Demo HTML frontend |
| `GET`  | `/health` | Model load status |
| `GET`  | `/scenarios` | Available degradation scenarios |
| `POST` | `/restore` | Restore diacritics; returns text + per-character confidence |
| `POST` | `/degrade` | Degrade clean text (data-augmentation pipeline) |
| `POST` | `/generate-dataset` | Generate full (input, target) JSONL dataset |
| `POST` | `/postprocess` | Apply the rule-based post-processor only |

Full OpenAPI schema at `http://localhost:8000/docs` when the server is running.

## Limitations

- **Foreign place names** are the dominant out-of-distribution failure mode. The
  model was trained on Romanian news text (RoLargeSum) which rarely contains the
  older Romanian transliteration conventions used in Wikipedia articles about
  Eastern European geography (`Koșecikî`, `Lîstvîn`, `Jîtomîr` style names).
  CER on the OOD gold set is ~3× higher than in-distribution as a result. See
  §8.6 of the project report for examples.
- **OCR-noisy text triggers over-correction.** On out-of-distribution OCR-degraded
  text, the model achieves 100% recall but only 31% precision in change detection
  — it flags many already-correct characters as needing fixes.
- **Sentence-level training** means multi-paragraph inputs are split on newlines
  before being sent to the model; each line is processed independently. Inter-line
  context isn't used.

## Built with

PyTorch, HuggingFace Transformers, FastAPI, Pydantic v2, NumPy, Matplotlib.

Trained on RoLargeSum: [Avram et al., 2022](https://huggingface.co/datasets/avramandrei/RoLargeSum).
Model architecture: [ByT5 by Xue et al., 2022](https://arxiv.org/abs/2105.13626).

## License

MIT for the code. The trained model weights inherit the license of the base
[google/byt5-base](https://huggingface.co/google/byt5-base) (Apache 2.0). RoLargeSum
training data has its own license - consult the dataset card before use.