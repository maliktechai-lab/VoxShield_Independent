"""
VoxShield FastAPI Backend
==========================

Endpoints:
    GET  /health          — service liveness + model status
    GET  /model-info      — detailed model metadata
    POST /predict         — upload audio, run inference, log incident
    GET  /incidents       — list logged incidents (paginated)
    GET  /incidents/stats — aggregate statistics
    POST /reload-model    — reload model from disk (e.g., after training)

Run:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

Security:
    - No external inference APIs.
    - All processing is local.
    - Upload size limited to MAX_UPLOAD_MB.
    - Temporary files are always cleaned up.
    - No secrets exposed in responses.
    - CORS configured for local development.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional

# ── Project root on PYTHONPATH ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager
from fastapi import (
    Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from inference.engine import get_engine, reload_engine
from inference.preprocessor import AudioPreprocessor, AudioPreprocessorError
from backend.incident_logger import get_incident_logger
from backend.security import require_api_key

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

MAX_UPLOAD_MB = float(os.environ.get("VOXSHIELD_MAX_UPLOAD_MB", "25"))
ALLOWED_EXTENSIONS = {".flac", ".wav", ".ogg", ".mp3", ".m4a"}
APP_VERSION = os.environ.get("VOXSHIELD_API_VERSION", "1.1.0")
EXPOSE_PATHS = os.environ.get("VOXSHIELD_EXPOSE_PATHS", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "VOXSHIELD_CORS_ORIGINS", "http://localhost:5173"
    ).split(",")
    if origin.strip()
]
MAX_CONCURRENT_INFERENCES = max(
    1, int(os.environ.get("VOXSHIELD_MAX_CONCURRENT_INFERENCES", "2"))
)
INFERENCE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voxshield.backend")

# ── Lazy-init singletons ──────────────────────────────────────────────────────

_preprocessor: Optional[AudioPreprocessor] = None
_engine        = None
_incident_log  = None


@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Initialize singletons on startup."""
    global _preprocessor, _engine, _incident_log
    _preprocessor = AudioPreprocessor()
    _engine       = get_engine()
    _incident_log = get_incident_logger()
    logger.info(
        f"VoxShield backend started | "
        f"Model ready: {_engine.is_ready()} | "
        f"Mode: {_engine.inference_mode}"
    )
    yield
    # Shutdown — nothing to clean up


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VoxShield API",
    description=(
        "Local voice anti-spoofing detection. "
        "All inference runs on-device. No external AI APIs. "
        "VoxShieldNet is trained from random initialization."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — permissive for local development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _get_engine():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def _get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = AudioPreprocessor()
    return _preprocessor


def _get_logger():
    global _incident_log
    if _incident_log is None:
        _incident_log = get_incident_logger()
    return _incident_log


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Service health check")
async def health():
    """
    Returns liveness + model readiness.

    Does NOT return model metrics, thresholds, or checkpoint paths
    (those are in /model-info).
    """
    engine = _get_engine()
    return {
        "status": "ok",
        "model_ready": engine.is_ready(),
        "device": str(engine.device),
        "inference_mode": engine.inference_mode,
        "version": APP_VERSION,
    }


@app.get("/ready", summary="Model readiness check")
async def ready():
    engine = _get_engine()
    if not engine.is_ready():
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ready", "model_ready": True, "version": APP_VERSION}


@app.get(
    "/model-info",
    summary="Detailed model information",
    dependencies=[Depends(require_api_key)],
)
async def model_info():
    """
    Returns complete model metadata including:
    - Architecture details
    - Checkpoint location
    - Dataset info (ASVspoof5 vs demo)
    - Validation metrics (from checkpoint — NOT fabricated)
    - Decision threshold (from EER on validation set)
    - pretrained=false (always)

    IMPORTANT: Metrics shown here are from the training checkpoint.
    They reflect performance on the validation split, NOT a held-out test set.
    Do not interpret these as production accuracy figures.
    """
    engine = _get_engine()
    info = engine.model_info()
    # Explicitly mark pretrained=false at API level.
    info["pretrained"] = False
    info["external_api"] = False
    if not EXPOSE_PATHS:
        info["checkpoint"] = None
        info["dataset_source"] = None
    return info


@app.post(
    "/predict", summary="Analyze uploaded audio for spoofing",
    dependencies=[Depends(require_api_key)],
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Audio file (.flac, .wav, .ogg, .mp3)"),
    session_id: Optional[str] = Form(None),
):
    """
    Upload an audio file and receive a spoof detection result.

    The result includes:
    - classification (SPOOF | BONA_FIDE)
    - spoof_probability and bona_fide_probability
    - confidence
    - decision_threshold (from trained checkpoint)
    - risk_score, threat_level, recommended_action
    - latency_ms, device, inference_mode

    When no trained model exists:
    Returns status=unavailable with a clear message.
    Never returns fabricated predictions.
    """
    engine = _get_engine()
    preprocessor = _get_preprocessor()
    incident_log = _get_logger()

    sid = session_id or str(uuid.uuid4())[:8]
    filename = file.filename or "unknown"

    # ── File size check ───────────────────────────────────────────────────────
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > int(MAX_UPLOAD_MB * 1024 * 1024 * 1.10):
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds maximum size of {MAX_UPLOAD_MB:.0f} MB.",
                )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.")

    raw_bytes = await file.read()
    file_size = len(raw_bytes)
    try:
        preprocessor.validate_upload_size(file_size, max_mb=MAX_UPLOAD_MB)
    except AudioPreprocessorError as e:
        raise HTTPException(status_code=413, detail=str(e))

    # ── Extension check ───────────────────────────────────────────────────────
    suffix = Path(filename).suffix.lower()
    if not suffix:
        suffix = ".wav"
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{suffix}'. "
                f"Accepted: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    # ── Audio preprocessing ───────────────────────────────────────────────────
    try:
        wav = await run_in_threadpool(preprocessor.from_bytes, raw_bytes, suffix)
    except AudioPreprocessorError as e:
        result = {
            "classification":         "ERROR",
            "spoof_probability":      None,
            "bona_fide_probability":  None,
            "confidence":             None,
            "decision_threshold":     engine.threshold,
            "risk_score":             None,
            "threat_level":           "UNKNOWN",
            "recommended_action":     "Audio could not be decoded. Check file format.",
            "risk_breakdown":         {},
            "latency_ms":             0.0,
            "device":                 str(engine.device),
            "model_version":          None,
            "inference_mode":         engine.inference_mode,
            "status":                 "error",
            "error":                  str(e),
        }
        await run_in_threadpool(
            incident_log.log_incident,
            result,
            filename=filename,
            file_size_bytes=file_size,
            session_id=sid,
        )
        raise HTTPException(status_code=422, detail=str(e))

    # ── Inference ─────────────────────────────────────────────────────────────
    async with INFERENCE_SEMAPHORE:
        result = await run_in_threadpool(engine.predict, wav)

    # ── Logging ───────────────────────────────────────────────────────────────
    await run_in_threadpool(
        incident_log.log_incident,
        result,
        filename=filename,
        file_size_bytes=file_size,
        session_id=sid,
    )

    # ── Response ──────────────────────────────────────────────────────────────
    return JSONResponse(content=result)


@app.get(
    "/incidents", summary="List logged incidents",
    dependencies=[Depends(require_api_key)],
)
async def get_incidents(
    limit: int = 50,
    offset: int = 0,
    classification: Optional[str] = None,
    threat_level: Optional[str] = None,
):
    """Return recent incidents from the local SQLite log."""
    if limit > 500:
        limit = 500
    incident_log = _get_logger()
    incidents = incident_log.get_incidents(
        limit=limit,
        offset=offset,
        classification=classification,
        threat_level=threat_level,
    )
    return {"incidents": incidents, "count": len(incidents), "offset": offset}


@app.get(
    "/incidents/stats", summary="Incident aggregate statistics",
    dependencies=[Depends(require_api_key)],
)
async def incidents_stats():
    """Return aggregate statistics over all logged incidents."""
    incident_log = _get_logger()
    return incident_log.get_stats()


@app.get(
    "/incidents/timeline", summary="Recent incident timeline (for dashboard)",
    dependencies=[Depends(require_api_key)],
)
async def incidents_timeline(n: int = 20):
    """Return last N incidents for live timeline display."""
    if n > 100:
        n = 100
    incident_log = _get_logger()
    return {"timeline": incident_log.get_recent_timeline(n)}


@app.post(
    "/reload-model", summary="Reload model from disk",
    dependencies=[Depends(require_api_key)],
)
async def reload_model():
    """
    Reload the inference model from disk.
    Call this after training finishes to pick up the new checkpoint
    without restarting the server.
    """
    global _engine
    _engine = reload_engine()
    return {
        "status":      "reloaded",
        "model_ready": _engine.is_ready(),
        "checkpoint":  str(_engine.checkpoint_path) if _engine.checkpoint_path else None,
        "inference_mode": _engine.inference_mode,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal server error",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
