"""
FastAPI service — Data Generation + Diacritics Restoration
Project: Diacritizare robustă și detecție de diacritice greșite (română)

Endpoints:
  GET  /                 — demo frontend (HTML page)
  POST /restore          — restore diacritics using trained ByT5 model  ← NEW
  POST /degrade          — degrade a single text or batch of texts
  POST /generate-dataset — generate a full (input, target) dataset from a source
  GET  /scenarios        — list available degradation scenarios
  GET  /health           — health check (reports model status)
  POST /postprocess      — run post-processing on a model output
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field, field_validator

from degradation import (
    DegradationConfig,
    DegradationScenario,
    DegradedSample,
    degrade_text,
    generate_samples,
)
from data_loader import (
    DEMO_SENTENCES,
    load_text_file,
    load_rolargesum,
    split_sentences,
)
from postprocessor import postprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diacritics model — optional import (so data-generation endpoints still work
# even if torch/transformers aren't installed or the trained model is missing)
# ---------------------------------------------------------------------------

try:
    from predict import DiacriticsRestorer, RestorationResult
    _PREDICT_AVAILABLE = True
except ImportError as e:
    _PREDICT_AVAILABLE = False
    _PREDICT_IMPORT_ERROR = str(e)
    logger.warning("Diacritics model module not available: %s", e)

# Where to load the trained model from. Default assumes uvicorn is launched
# from the `app/` directory (as is already the convention).
_MODEL_PATH = os.environ.get(
    "DIACRITICS_MODEL_PATH",
    "../models/byt5-diacritics/final",
)

# Populated by the lifespan startup hook, read by the /restore endpoint.
_RESTORER: Optional["DiacriticsRestorer"] = None


# ---------------------------------------------------------------------------
# App lifecycle: load the model once at startup, keep it in memory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained ByT5 model into memory once, at server startup."""
    global _RESTORER

    if not _PREDICT_AVAILABLE:
        logger.warning(
            "Diacritics restoration disabled — `predict` module not importable "
            "(%s). /restore endpoint will return 503.",
            _PREDICT_IMPORT_ERROR,
        )
    elif not os.path.exists(_MODEL_PATH):
        logger.warning(
            "Trained model not found at %s — /restore endpoint will return 503. "
            "Set DIACRITICS_MODEL_PATH env var or train the model first.",
            _MODEL_PATH,
        )
    else:
        try:
            logger.info("Loading diacritics restoration model from %s ...", _MODEL_PATH)
            t0 = time.perf_counter()
            _RESTORER = DiacriticsRestorer(_MODEL_PATH)
            logger.info("Model loaded in %.1fs.", time.perf_counter() - t0)
        except Exception as e:
            logger.exception("Failed to load model from %s: %s", _MODEL_PATH, e)
            _RESTORER = None

    yield
    # (no cleanup needed — PyTorch/transformers release GPU memory on process exit)


app = FastAPI(
    title="Romanian Diacritics — Restoration & Data Generation API",
    description=(
        "Restores diacritics on Romanian text using a fine-tuned ByT5 model, "
        "and generates degraded training data. Part of the NLP Techniques project (FII UAIC)."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RestoreRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000,
                      description="Romanian text (may have missing or wrong diacritics).")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
        return v


class RestoreResponse(BaseModel):
    input: str
    restored: str
    changed_positions: list[int]
    confidence: float
    postprocessing_changes: list[str]
    elapsed_ms: float


class DegradeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="One or more clean Romanian sentences.")
    scenario: DegradationScenario = DegradationScenario.MIXED
    strip_probability: float = Field(0.5, ge=0.0, le=1.0)
    wrong_diacritic_probability: float = Field(0.3, ge=0.0, le=1.0)
    wrong_swap_probability: float = Field(0.2, ge=0.0, le=1.0)
    ocr_probability: float = Field(0.02, ge=0.0, le=1.0)
    per_text_variants: int = Field(1, ge=1, le=10, description="Degraded variants per input text.")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility.")

    @field_validator("texts")
    @classmethod
    def texts_not_empty(cls, v: list[str]) -> list[str]:
        if any(not t.strip() for t in v):
            raise ValueError("texts must not contain empty strings")
        return v


class DegradedSampleOut(BaseModel):
    input: str
    target: str
    scenario: str
    changed_positions: list[int]


class DegradeResponse(BaseModel):
    samples: list[DegradedSampleOut]
    total: int
    elapsed_ms: float


class DatasetRequest(BaseModel):
    source: str = Field(
        "demo",
        description=(
            "Data source. Options: 'demo' | 'rolargesum' | 'raw'. "
            "Use 'raw' together with the `texts` field to supply your own sentences."
        ),
    )
    texts: Optional[list[str]] = Field(None, description="Used when source='raw'.")
    max_sentences: int = Field(1000, ge=1, le=100_000)
    per_text_variants: int = Field(1, ge=1, le=5)
    scenario: DegradationScenario = DegradationScenario.MIXED
    strip_probability: float = Field(0.5, ge=0.0, le=1.0)
    wrong_diacritic_probability: float = Field(0.3, ge=0.0, le=1.0)
    wrong_swap_probability: float = Field(0.2, ge=0.0, le=1.0)
    ocr_probability: float = Field(0.02, ge=0.0, le=1.0)
    seed: Optional[int] = 42
    output_format: str = Field("json", description="'json' or 'jsonl'")

    @field_validator("output_format")
    @classmethod
    def valid_format(cls, v: str) -> str:
        if v not in ("json", "jsonl"):
            raise ValueError("output_format must be 'json' or 'jsonl'")
        return v


class PostprocessRequest(BaseModel):
    original_text: str = Field(..., description="The original degraded text that was fed to the model.")
    model_output: str = Field(..., description="The raw text the model produced.")


class PostprocessResponse(BaseModel):
    original_output: str
    final_text: str
    changes_made: list[str]
    was_modified: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_config(req: DegradeRequest | DatasetRequest) -> DegradationConfig:
    return DegradationConfig(
        scenario=req.scenario,
        strip_probability=req.strip_probability,
        wrong_diacritic_probability=req.wrong_diacritic_probability,
        wrong_swap_probability=req.wrong_swap_probability,
        ocr_probability=req.ocr_probability,
        seed=req.seed if hasattr(req, "seed") else None,
    )


def _sample_to_out(s: DegradedSample) -> DegradedSampleOut:
    return DegradedSampleOut(
        input=s.input,
        target=s.target,
        scenario=s.scenario,
        changed_positions=s.changed_positions,
    )


def _load_source(req: DatasetRequest) -> list[str]:
    source = req.source.lower()
    if source == "demo":
        return DEMO_SENTENCES
    elif source == "rolargesum":
        return load_rolargesum(max_sentences=req.max_sentences)
    elif source == "raw":
        if not req.texts:
            raise HTTPException(status_code=422, detail="Provide `texts` when source='raw'.")
        sentences: list[str] = []
        for t in req.texts:
            sentences.extend(split_sentences(t) or [t])
        return sentences[: req.max_sentences]
    else:
        raise HTTPException(status_code=422, detail=f"Unknown source '{source}'.")


# Simple sentence splitter for chunking long /restore inputs. Unlike
# data_loader.split_sentences it does NOT filter by length — we want every
# sentence, not only the "training-worthy" ones.
_RESTORE_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Byte budget per model call. ByT5 was fine-tuned at max_input_length=512;
# we stay a bit under to leave headroom for the tokenizer's special tokens.
_RESTORE_BYTE_BUDGET = 400


def _restore_long_text(text: str, restorer: "DiacriticsRestorer") -> tuple[str, float]:
    """Restore diacritics on text of any length by chunking into sentences.

    Returns (restored_text, mean_confidence). Chunking is only triggered when
    the input exceeds the model's comfortable byte budget — short inputs go
    through a single model call.
    """
    if len(text.encode("utf-8")) <= _RESTORE_BYTE_BUDGET:
        r = restorer.restore(text)
        return r.restored, r.confidence

    parts = _RESTORE_SENT_SPLIT_RE.split(text.strip())
    restored_parts: list[str] = []
    confidences: list[float] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        r = restorer.restore(part)
        restored_parts.append(r.restored)
        confidences.append(r.confidence)

    restored = " ".join(restored_parts)
    mean_conf = sum(confidences) / len(confidences) if confidences else 1.0
    return restored, mean_conf


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def frontend():
    """Serve the demo HTML frontend."""
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _RESTORER is not None,
        "model_path": _MODEL_PATH if _RESTORER is not None else None,
    }


@app.post("/restore", response_model=RestoreResponse)
def restore(req: RestoreRequest):
    """
    Restore diacritics on Romanian text.

    Runs the fine-tuned ByT5 model, then applies the rule-based post-processor
    (URL/number/acronym preservation, cedilla fixes, OCR cleanup), and returns
    the restored text plus metadata for the demo UI.
    """
    if _RESTORER is None:
        detail = (
            "Trained model is not loaded. "
            "Train the model first (see scripts/train.py) or set the "
            "DIACRITICS_MODEL_PATH environment variable to a valid model directory."
        )
        raise HTTPException(status_code=503, detail=detail)

    t0 = time.perf_counter()

    # 1) Model restoration (with auto-chunking for long inputs)
    model_restored, confidence = _restore_long_text(req.text, _RESTORER)

    # 2) Rule-based post-processor (URL/number/acronym preservation, cedilla, OCR)
    pp = postprocess(req.text, model_restored)
    final_text = pp.final_text

    # 3) Recompute changed positions against the final post-processed text so
    #    the UI highlights match what the user actually sees.
    changed_positions = DiacriticsRestorer._diff_positions(req.text, final_text)

    elapsed = (time.perf_counter() - t0) * 1000

    return RestoreResponse(
        input=req.text,
        restored=final_text,
        changed_positions=changed_positions,
        confidence=confidence,
        postprocessing_changes=pp.changes_made,
        elapsed_ms=round(elapsed, 2),
    )


@app.get("/scenarios")
def list_scenarios():
    """Return all available degradation scenarios with descriptions."""
    return {
        "scenarios": [
            {
                "id": DegradationScenario.STRIP_ALL,
                "description": "Remove ALL diacritics (ș→s, ț→t, ă→a, â→a, î→i).",
            },
            {
                "id": DegradationScenario.STRIP_PARTIAL,
                "description": "Remove diacritics randomly per character (strip_probability controls rate).",
            },
            {
                "id": DegradationScenario.WRONG_DIACRITICS,
                "description": "Introduce wrong diacritics: swap existing or add incorrect ones on base chars.",
            },
            {
                "id": DegradationScenario.CEDILLA,
                "description": "Replace correct ș/ț (comma-below) with legacy ş/ţ (cedilla variants).",
            },
            {
                "id": DegradationScenario.OCR,
                "description": "OCR-style noise: character confusions (1↔l, 0↔o, rn↔m) + space split/merge.",
            },
            {
                "id": DegradationScenario.MIXED,
                "description": "Randomly pick a scenario per sample (recommended for training data).",
            },
        ]
    }


@app.post("/degrade", response_model=DegradeResponse)
def degrade(req: DegradeRequest):
    """
    Degrade one or more clean Romanian texts and return (input, target) pairs.
    Useful for quick experimentation or integration testing.
    """
    t0 = time.perf_counter()
    config = _build_config(req)
    samples = generate_samples(req.texts, config, per_text_variants=req.per_text_variants)
    elapsed = (time.perf_counter() - t0) * 1000

    return DegradeResponse(
        samples=[_sample_to_out(s) for s in samples],
        total=len(samples),
        elapsed_ms=round(elapsed, 2),
    )


@app.post("/postprocess", response_model=PostprocessResponse)
def postprocess_endpoint(req: PostprocessRequest):
    """
    Run post-processing on a model output.

    Accepts the original degraded text and the raw model output, applies
    all post-processing rules, and returns the cleaned result.
    """
    result = postprocess(req.original_text, req.model_output)

    return PostprocessResponse(
        original_output=result.original_output,
        final_text=result.final_text,
        changes_made=result.changes_made,
        was_modified=result.was_modified,
    )


@app.post("/generate-dataset")
def generate_dataset(req: DatasetRequest):
    """
    Load text from a source, apply degradation, and return a dataset.

    For large datasets, use output_format='jsonl' — the response streams line-by-line
    so you can pipe it directly to a file:
      curl -X POST .../generate-dataset -d '...' | jq -c '.' > dataset.jsonl
    """
    texts = _load_source(req)

    if not texts:
        raise HTTPException(status_code=404, detail="No usable sentences found in source.")

    config = _build_config(req)

    if req.output_format == "jsonl":
        # Stream JSONL — memory-efficient for large datasets
        def _stream():
            for sample in generate_samples(texts, config, per_text_variants=req.per_text_variants):
                yield json.dumps(
                    {
                        "input": sample.input,
                        "target": sample.target,
                        "scenario": sample.scenario,
                        "changed_positions": sample.changed_positions,
                    },
                    ensure_ascii=False,
                ) + "\n"

        return StreamingResponse(
            _stream(),
            media_type="application/x-ndjson",
            headers={"X-Total-Sentences": str(len(texts))},
        )
    else:
        samples = generate_samples(texts, config, per_text_variants=req.per_text_variants)
        return JSONResponse(
            content={
                "source": req.source,
                "total_sentences": len(texts),
                "total_samples": len(samples),
                "samples": [
                    {
                        "input": s.input,
                        "target": s.target,
                        "scenario": s.scenario,
                        "changed_positions": s.changed_positions,
                    }
                    for s in samples
                ],
            }
        )


@app.post("/generate-dataset/stats")
def dataset_stats(req: DatasetRequest):
    """
    Same as /generate-dataset but returns only aggregate statistics — useful
    for quickly validating degradation coverage without downloading everything.
    """
    texts = _load_source(req)
    config = _build_config(req)
    samples = generate_samples(texts, config, per_text_variants=req.per_text_variants)

    scenario_counts: dict[str, int] = {}
    total_changed = 0
    total_chars = 0

    for s in samples:
        scenario_counts[s.scenario] = scenario_counts.get(s.scenario, 0) + 1
        total_changed += len(s.changed_positions)
        total_chars += len(s.target)

    return {
        "source": req.source,
        "total_sentences": len(texts),
        "total_samples": len(samples),
        "scenario_distribution": scenario_counts,
        "avg_changed_chars_per_sample": round(total_changed / max(len(samples), 1), 2),
        "avg_change_rate": round(total_changed / max(total_chars, 1), 4),
    }
