"""
api_fastapi.py (Phase 8 — optimised)
--------------------------------------
FastAPI application integrating:
  - resume parsing (Phase 1)
  - job collection (Phase 2)
  - hybrid scoring + ranking (Phases 4–7)
  - skill gap (Phase 10)

Startup optimisations
---------------------
* All ML models are loaded ONCE via the ``lifespan`` async context manager
  (the modern FastAPI replacement for deprecated ``@app.on_event("startup")``).
* ``get_model()`` is a FastAPI dependency that returns the already-loaded
  singleton from ModelRegistry — zero I/O on every request.
* The registry is accessed through ``model.ml_scorer.get_registry()`` so
  there is a single source of truth across Flask + FastAPI if both run in
  the same process.

Endpoints
---------
  POST /upload-resume  -> parsed resume JSON
  GET  /jobs           -> cached or live job list JSON
  POST /match          -> top matched jobs for uploaded resume
  GET  /health         -> status + model readiness check
"""

from __future__ import annotations

import json
import uuid
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Dict, List
import contextvars

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from collect_jobs import collect_jobs
from model.ml_scorer import (
    ModelRegistry,
    ensure_default_model,
    get_registry,
    warm_up,
)
from model.ranking_engine import rank_jobs
from model.resume_structured import parse_resume_pdf

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
request_id_var = contextvars.ContextVar("request_id", default="SYSTEM")

class FastApiRequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - [req:%(request_id)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.addFilter(FastApiRequestIdFilter())
logging.getLogger().addFilter(FastApiRequestIdFilter())

BASE_DIR  = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
JOBS_CACHE = BASE_DIR / "jobs_data.json"
RESUME_SAMPLE_OUT = BASE_DIR / "resume_data.json"

# ---------------------------------------------------------------------------
# Lifespan — load all models at startup, once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Warming up ML models...")
    warm_up(BASE_DIR / "model_store")
    logger.info("ML models warm-up complete.")
    yield  

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Resume Analyzer API",
    version="2.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def add_process_time_header_and_trace(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    request_id_var.set(req_id)
    start_time = time.time()
    
    logger.info(f"Started {request.method} {request.url.path}")
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logger.info(f"Completed {request.method} {request.url.path} - Status: {response.status_code} - {process_time:.2f}ms")
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = req_id
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dependency — returns the singleton registry (no I/O, just a dict lookup)
# ---------------------------------------------------------------------------

def get_model_registry() -> ModelRegistry:
    """
    FastAPI dependency that returns the already-loaded ModelRegistry.

    Because ``warm_up()`` was called in ``lifespan``, this is always
    instantaneous — no file I/O happens here.

    Raises HTTPException 503 if the score model failed to load at startup.
    """
    reg = get_registry()
    if not reg.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Scoring model is not available. "
                "Check server logs for startup errors."
            ),
        )
    return reg


# Annotated alias used in route signatures
RegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload-resume")
async def upload_resume(resume: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a PDF resume; return structured JSON:
      skills, experience, education, meta, raw_text (truncated)
    """
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    safe = f"{uuid.uuid4().hex}_{resume.filename}"
    path = UPLOAD_DIR / safe
    content = await resume.read()
    path.write_bytes(content)

    try:
        data = parse_resume_pdf(path)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Failed to parse PDF: {exc}"
        ) from exc

    # Persist latest sample for local inspection
    try:
        RESUME_SAMPLE_OUT.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

    return JSONResponse(content=data)


@app.get("/jobs")
def get_jobs(refresh: bool = False) -> JSONResponse:
    """
    Return job listings as JSON.
    Uses cached ``jobs_data.json`` unless ``refresh=true``.
    """
    if refresh or not JOBS_CACHE.is_file():
        try:
            collect_jobs(out_path=JOBS_CACHE, limit=30, prefer_api=True)
        except Exception:
            collect_jobs(out_path=JOBS_CACHE, limit=50, prefer_api=False)

    try:
        jobs = json.loads(JOBS_CACHE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not read jobs file: {exc}"
        ) from exc

    return JSONResponse(content=jobs)


@app.post("/match")
async def match_resume(
    registry: RegistryDep,
    resume: UploadFile = File(...),
    top_n: int = 10,
    include_debug: bool = False,
) -> JSONResponse:
    """
    Upload resume PDF; return top N ranked jobs with scores and missing skills.

    The scoring model is injected as a FastAPI dependency — it was loaded once
    at startup and is reused across all requests with zero additional I/O.
    """
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    safe = f"{uuid.uuid4().hex}_{resume.filename}"
    path = UPLOAD_DIR / safe
    path.write_bytes(await resume.read())

    logger.info(f"Received match request for {resume.filename}")
    try:
        start_time = time.time()
        resume_data = parse_resume_pdf(path)
        logger.info(f"Resume parsed in {(time.time() - start_time)*1000:.2f}ms")
    except Exception as exc:
        logger.error(f"Failed to parse PDF: {exc}")
        raise HTTPException(
            status_code=422, detail=f"Failed to parse PDF: {exc}"
        ) from exc

    if not JOBS_CACHE.is_file():
        collect_jobs(out_path=JOBS_CACHE, limit=50, prefer_api=False)

    try:
        jobs: List[Dict[str, Any]] = json.loads(
            JOBS_CACHE.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not read jobs: {exc}"
        ) from exc

    # Rank jobs (semantic + experience) without ML model dependency
    start_time = time.time()
    ranked = rank_jobs(resume_data, jobs, top_n=min(max(top_n, 1), 50))
    logger.info(f"Ranked {len(jobs)} jobs in {(time.time() - start_time)*1000:.2f}ms. Top score: {ranked[0]['score'] if ranked else 0}%")

    if not include_debug:
        for row in ranked:
            row.pop("debug", None)
            row.pop("feature_vector", None)

    return JSONResponse(
        content={
            "resume": {
                "skills":     resume_data.get("skills"),
                "experience": resume_data.get("experience"),
                "education":  resume_data.get("education"),
            },
            "top_jobs": ranked,
        }
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check — reports model readiness without raising exceptions."""
    reg = get_registry()
    return {
        "status": "ok" if reg.is_ready() else "degraded",
        "models": reg.status(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_fastapi:app", host="127.0.0.1", port=8000, reload=True)
