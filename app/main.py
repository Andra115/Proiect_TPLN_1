"""
FastAPI service — Data Generation + Diacritics Restoration + Detection
Project: Diacritizare robustă și detecție de diacritice greșite (română)

Endpoints:
  GET  /                      — demo frontend (HTML page)
  GET  /health                — health check (reports model + detector status)
  GET  /scenarios             — list available degradation scenarios

  POST /restore               — restore diacritics using trained ByT5 model (§3.2)
  POST /detect                — detect changed positions + entropy scores (§3.3)
  POST /detect-files          — same, reads from file paths
  POST /restore-and-detect    — restore then immediately detect in one call (§3.2 + §3.3)
  GET  /detect/threshold      — explain current review threshold

  POST /degrade               — degrade a single text or batch of texts (§3.1)
  POST /generate-dataset      — generate a full (input, target) dataset (§3.1)
  POST /generate-dataset/stats— stats-only variant
  POST /postprocess           — run post-processing on a model output
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Path setup — must be first so all relative imports resolve correctly
# ---------------------------------------------------------------------------
from pathlib import Path
_APP_DIR = Path(__file__).parent

import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field, field_validator

# All sibling modules use relative imports since this file lives inside the
# `app` package and uvicorn is launched with `uvicorn app.main:app`.
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
    load_rolargesum_local,
    split_sentences,
)
from postprocessor import postprocess
from detection.detector import Detector, DetectionReport, build_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diacritics model — optional import
# ---------------------------------------------------------------------------

try:
    from predict import DiacriticsRestorer, RestorationResult
    _PREDICT_AVAILABLE = True
except ImportError as e:
    _PREDICT_AVAILABLE = False
    _PREDICT_IMPORT_ERROR = str(e)
    logger.warning("Diacritics model module not available: %s", e)

_MODEL_PATH = os.environ.get(
    "DIACRITICS_MODEL_PATH",
    str(_APP_DIR.parent / "models" / "byt5-diacritics" / "final"),
)

# Populated by lifespan startup hook
_RESTORER: Optional["DiacriticsRestorer"] = None
_REVIEW_THRESHOLD: float = 1.0
_detector: Optional[Detector] = None


# ---------------------------------------------------------------------------
# Lifespan: load model + fit detector LM once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _RESTORER, _detector

    # --- §3.2 model loading ---
    if not _PREDICT_AVAILABLE:
        logger.warning(
            "Diacritics restoration disabled — `predict` module not importable (%s). "
            "/restore endpoint will return 503.",
            _PREDICT_IMPORT_ERROR,
        )
    elif not os.path.exists(_MODEL_PATH):
        logger.warning(
            "Trained model not found at %s — /restore will return 503. "
            "Set DIACRITICS_MODEL_PATH or train the model first.",
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

    # --- §3.3 detector: fit n-gram LM on local RoLargeSum data ---
    try:
        logger.info("Loading RoLargeSum for detector n-gram LM…")
        texts = load_rolargesum_local(max_sentences=50_000)
        _detector = build_detector(texts, review_threshold=_REVIEW_THRESHOLD)
        logger.info("Detector LM fitted on %d sentences.", len(texts))
    except Exception as exc:
        logger.warning("Detector startup failed (%s); falling back to demo sentences.", exc)
        _detector = build_detector(DEMO_SENTENCES, review_threshold=_REVIEW_THRESHOLD)

    yield
    # No cleanup needed — memory released on process exit


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Romanian Diacritics — Restoration & Data Generation API",
    description=(
        "Restores diacritics on Romanian text using a fine-tuned ByT5 model, "
        "detects uncertain corrections, and generates degraded training data. "
        "Part of the NLP Techniques project (FII UAIC)."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(_APP_DIR / "static")), name="static")


# ===========================================================================
# Schemas — ALL defined here, before any endpoint that references them
# ===========================================================================

# --- Restore ---

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


# --- Detection ---

class DetectRequest(BaseModel):
    input_text: str = Field(
        ...,
        description="The original degraded text (no diacritics / wrong diacritics).",
    )
    output_text: str = Field(
        ...,
        description="The corrected text produced by the restoration model (§3.2).",
    )
    review_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description=(
            "Shannon entropy threshold (bits) above which a change is flagged for "
            "human review. Overrides the server default for this request only."
        ),
    )


class DetectFilesRequest(BaseModel):
    input_path: str = Field(..., description="Path to the degraded input text file.")
    output_path: str = Field(..., description="Path to the corrected output text file.")
    review_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class DetectionResultOut(BaseModel):
    position: int
    original_char: str
    corrected_char: str
    context: str
    change_type: str
    probability: float
    entropy: float
    flag_for_review: bool
    scoring_method: str


class DetectResponse(BaseModel):
    input_text: str
    corrected_text: str
    total_changes: int
    flagged_count: int
    avg_entropy: float
    elapsed_ms: float
    detections: list[DetectionResultOut]


# --- Restore + Detect combined ---

class RestoreAndDetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000,
                      description="Romanian text (may have missing or wrong diacritics).")
    review_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Entropy threshold for flagging. Defaults to server setting.",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
        return v


class RestoreAndDetectResponse(BaseModel):
    """Combined response: restoration result + per-character detection scores."""
    # Restoration fields
    input: str
    restored: str
    confidence: float
    postprocessing_changes: list[str]
    # Detection fields
    total_changes: int
    flagged_count: int
    avg_entropy: float
    detections: list[DetectionResultOut]
    # Timing broken down so you can see the cost of each stage
    restore_ms: float
    detect_ms: float
    total_ms: float


# --- Degrade / Dataset ---

class DegradeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1,
                             description="One or more clean Romanian sentences.")
    scenario: DegradationScenario = DegradationScenario.MIXED
    strip_probability: float = Field(0.5, ge=0.0, le=1.0)
    wrong_diacritic_probability: float = Field(0.3, ge=0.0, le=1.0)
    wrong_swap_probability: float = Field(0.2, ge=0.0, le=1.0)
    ocr_probability: float = Field(0.02, ge=0.0, le=1.0)
    per_text_variants: int = Field(1, ge=1, le=10)
    seed: Optional[int] = Field(None)

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
    original_text: str = Field(..., description="The original degraded text.")
    model_output: str = Field(..., description="The raw text the model produced.")


class PostprocessResponse(BaseModel):
    original_output: str
    final_text: str
    changes_made: list[str]
    was_modified: bool


# ===========================================================================
# Helpers
# ===========================================================================

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
        input=s.input, target=s.target,
        scenario=s.scenario, changed_positions=s.changed_positions,
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


def _get_detector(threshold: Optional[float]) -> Detector:
    """Return the global detector, or a threshold-overridden copy if requested."""
    if threshold is None or threshold == _REVIEW_THRESHOLD:
        return _detector
    from .detection.scorer import DiacriticScorer
    scorer = DiacriticScorer(review_threshold=threshold)
    scorer._lm = _detector._scorer._lm
    scorer._fitted = _detector._scorer._fitted
    return Detector(scorer)


def _report_to_detections(report: DetectionReport) -> list[DetectionResultOut]:
    return [
        DetectionResultOut(
            position=d.position,
            original_char=d.original_char,
            corrected_char=d.corrected_char,
            context=d.context,
            change_type=d.change_type,
            probability=d.probability,
            entropy=d.entropy,
            flag_for_review=d.flag_for_review,
            scoring_method=d.scoring_method,
        )
        for d in report.detections
    ]


def _report_to_response(report: DetectionReport, elapsed_ms: float) -> DetectResponse:
    return DetectResponse(
        input_text=report.input_text,
        corrected_text=report.corrected_text,
        total_changes=report.total_changes,
        flagged_count=report.flagged_count,
        avg_entropy=report.avg_entropy,
        elapsed_ms=round(elapsed_ms, 2),
        detections=_report_to_detections(report),
    )


_RESTORE_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
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

# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/", response_class=HTMLResponse)
def frontend():
    index = _APP_DIR / "static" / "index.html"
    if not index.exists():
        return HTMLResponse("<h2>No frontend yet — use <a href='/docs'>/docs</a></h2>")
    return FileResponse(str(index))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _RESTORER is not None,
        "detector_loaded": _detector is not None,
        "model_path": _MODEL_PATH if _RESTORER is not None else None,
    }


# --- §3.2 ---

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


# --- §3.3 ---

@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    """
    Detect changed positions between degraded input and corrected output,
    scoring each change with Shannon entropy from a Romanian character n-gram LM.
    """
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialised yet.")
    if not req.input_text.strip():
        raise HTTPException(status_code=422, detail="input_text must not be empty.")
    if not req.output_text.strip():
        raise HTTPException(status_code=422, detail="output_text must not be empty.")

    t0 = time.perf_counter()
    report = _get_detector(req.review_threshold).run(req.input_text, req.output_text)
    elapsed = (time.perf_counter() - t0) * 1000
    return _report_to_response(report, elapsed)


@app.post("/detect-files", response_model=DetectResponse)
def detect_files(req: DetectFilesRequest):
    """Same as /detect but reads input/output from file paths."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialised yet.")

    input_path, output_path = Path(req.input_path), Path(req.output_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"input_path not found: {req.input_path}")
    if not output_path.exists():
        raise HTTPException(status_code=404, detail=f"output_path not found: {req.output_path}")
    try:
        input_text = input_path.read_text(encoding="utf-8")
        output_text = output_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read files: {exc}")

    t0 = time.perf_counter()
    report = _get_detector(req.review_threshold).run(input_text, output_text)
    elapsed = (time.perf_counter() - t0) * 1000
    return _report_to_response(report, elapsed)


@app.get("/detect/threshold")
def get_threshold():
    return {
        "current_threshold_bits": _REVIEW_THRESHOLD,
        "explanation": {
            "entropy_0_to_0.5": "High confidence — n-gram strongly prefers the corrected character.",
            "entropy_0.5_to_1.0": "Moderate confidence — some ambiguity remains.",
            "entropy_above_1.0": "Low confidence — flagged for human review.",
        },
        "confusion_sets": {
            "ș/s/ş": "strip_all, cedilla, wrong_diacritic scenarios",
            "ț/t/ţ": "strip_all, cedilla, wrong_diacritic scenarios",
            "ă/â/a": "highest entropy — context is critical",
            "î/i": "moderate entropy",
        },
        "upgrade_path": (
            "When §3.2 logits are exposed per character, replace "
            "DiacriticScorer.score_change() with model softmax probabilities. "
            "All downstream logic stays the same."
        ),
    }


# --- §3.2 + §3.3 combined ---

@app.post("/restore-and-detect", response_model=RestoreAndDetectResponse)
def restore_and_detect(req: RestoreAndDetectRequest):
    """
    **§3.2 + §3.3 combined**

    Restores diacritics with the ByT5 model, then immediately runs detection
    on the result — returning restoration metadata and per-character entropy
    scores in a single call.

    Useful for the demo UI: one request gives everything needed to render
    the corrected text with highlighted uncertain positions.
    """
    if _RESTORER is None:
        raise HTTPException(status_code=503, detail=(
            "Trained model is not loaded. Train the model first or set "
            "DIACRITICS_MODEL_PATH to a valid model directory."
        ))
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialised yet.")

    # --- §3.2: Restore ---
    t0 = time.perf_counter()
    model_restored, confidence = _restore_long_text(req.text, _RESTORER)
    pp = postprocess(req.text, model_restored)
    final_text = pp.final_text
    restore_ms = (time.perf_counter() - t0) * 1000

    # --- §3.3: Detect on the restored output ---
    t1 = time.perf_counter()
    report = _get_detector(req.review_threshold).run(req.text, final_text)
    detect_ms = (time.perf_counter() - t1) * 1000

    return RestoreAndDetectResponse(
        input=req.text,
        restored=final_text,
        confidence=confidence,
        postprocessing_changes=pp.changes_made,
        total_changes=report.total_changes,
        flagged_count=report.flagged_count,
        avg_entropy=report.avg_entropy,
        detections=_report_to_detections(report),
        restore_ms=round(restore_ms, 2),
        detect_ms=round(detect_ms, 2),
        total_ms=round(restore_ms + detect_ms, 2),
    )


# --- §3.1 ---

@app.get("/scenarios")
def list_scenarios():
    return {
        "scenarios": [
            {"id": DegradationScenario.STRIP_ALL,
             "description": "Remove ALL diacritics (ș→s, ț→t, ă→a, â→a, î→i)."},
            {"id": DegradationScenario.STRIP_PARTIAL,
             "description": "Remove diacritics randomly per character."},
            {"id": DegradationScenario.WRONG_DIACRITICS,
             "description": "Introduce wrong diacritics."},
            {"id": DegradationScenario.CEDILLA,
             "description": "Replace ș/ț (comma-below) with legacy ş/ţ (cedilla)."},
            {"id": DegradationScenario.OCR,
             "description": "OCR-style noise: character confusions + space split/merge."},
            {"id": DegradationScenario.MIXED,
             "description": "Randomly pick a scenario per sample (recommended)."},
        ]
    }


@app.post("/degrade", response_model=DegradeResponse)
def degrade(req: DegradeRequest):
    """Degrade one or more clean Romanian texts and return (input, target) pairs."""
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
    """Apply rule-based post-processing to a model output."""
    result = postprocess(req.original_text, req.model_output)
    return PostprocessResponse(
        original_output=result.original_output,
        final_text=result.final_text,
        changes_made=result.changes_made,
        was_modified=result.was_modified,
    )


@app.post("/generate-dataset")
def generate_dataset(req: DatasetRequest):
    """Generate a degraded dataset. Use output_format='jsonl' for large sets."""
    texts = _load_source(req)
    if not texts:
        raise HTTPException(status_code=404, detail="No usable sentences found in source.")
    config = _build_config(req)

    if req.output_format == "jsonl":
        def _stream():
            for sample in generate_samples(texts, config, per_text_variants=req.per_text_variants):
                yield json.dumps({
                    "input": sample.input, "target": sample.target,
                    "scenario": sample.scenario,
                    "changed_positions": sample.changed_positions,
                }, ensure_ascii=False) + "\n"
        return StreamingResponse(
            _stream(), media_type="application/x-ndjson",
            headers={"X-Total-Sentences": str(len(texts))},
        )
    else:
        samples = generate_samples(texts, config, per_text_variants=req.per_text_variants)
        return JSONResponse(content={
            "source": req.source,
            "total_sentences": len(texts),
            "total_samples": len(samples),
            "samples": [
                {"input": s.input, "target": s.target,
                 "scenario": s.scenario, "changed_positions": s.changed_positions}
                for s in samples
            ],
        })


@app.post("/generate-dataset/stats")
def dataset_stats(req: DatasetRequest):
    """Return aggregate statistics for a degraded dataset without downloading it."""
    texts = _load_source(req)
    config = _build_config(req)
    samples = generate_samples(texts, config, per_text_variants=req.per_text_variants)
    scenario_counts: dict[str, int] = {}
    total_changed = total_chars = 0
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