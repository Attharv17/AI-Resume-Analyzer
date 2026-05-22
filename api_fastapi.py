"""
api_fastapi.py (Phase 8)
------------------------
FastAPI application integrating:
  - resume parsing (Phase 1)
  - job collection (Phase 2)
  - hybrid scoring + ranking (Phases 4–7)
  - skill gap (Phase 10)

Endpoints:
  POST /upload-resume  -> parsed resume JSON
  GET  /jobs           -> cached or live job list JSON
  POST /match          -> top matched jobs for uploaded resume
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from collect_jobs import collect_jobs
from model.ml_scorer import ensure_default_model
from model.ranking_engine import rank_jobs
from model.resume_structured import parse_resume_pdf

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
JOBS_CACHE = BASE_DIR / "jobs_data.json"
RESUME_SAMPLE_OUT = BASE_DIR / "resume_data.json"

app = FastAPI(title="AI Resume Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = ensure_default_model(BASE_DIR / "model_store")
    return _model


@app.post("/upload-resume")
async def upload_resume(resume: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a PDF resume; return structured JSON:
      skills, experience, education, meta, raw_text (truncated)
    """
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = f"{uuid.uuid4().hex}_{resume.filename}"
    path = UPLOAD_DIR / safe
    content = await resume.read()
    path.write_bytes(content)

    try:
        data = parse_resume_pdf(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {exc}") from exc

    # Also write latest sample to resume_data.json for local inspection
    try:
        RESUME_SAMPLE_OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass

    return JSONResponse(content=data)


@app.get("/jobs")
def get_jobs(refresh: bool = False) -> JSONResponse:
    """
    Return job listings as JSON. Uses cached `jobs_data.json` unless refresh=true.
    """
    if refresh or not JOBS_CACHE.is_file():
        try:
            collect_jobs(out_path=JOBS_CACHE, limit=30, prefer_api=True)
        except Exception:
            collect_jobs(out_path=JOBS_CACHE, limit=50, prefer_api=False)

    try:
        jobs = json.loads(JOBS_CACHE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read jobs file: {exc}") from exc

    return JSONResponse(content=jobs)


@app.post("/match")
async def match_resume(
    resume: UploadFile = File(...),
    top_n: int = 10,
    include_debug: bool = False,
) -> JSONResponse:
    """
    Upload resume PDF; return top N ranked jobs with scores and missing skills.
    """
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = f"{uuid.uuid4().hex}_{resume.filename}"
    path = UPLOAD_DIR / safe
    path.write_bytes(await resume.read())

    try:
        resume_data = parse_resume_pdf(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {exc}") from exc

    if not JOBS_CACHE.is_file():
        collect_jobs(out_path=JOBS_CACHE, limit=50, prefer_api=False)

    try:
        jobs: List[Dict[str, Any]] = json.loads(JOBS_CACHE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read jobs: {exc}") from exc

    model = get_model()
    ranked = rank_jobs(resume_data, jobs, model, top_n=min(max(top_n, 1), 50))

    if not include_debug:
        for row in ranked:
            row.pop("debug", None)
            row.pop("feature_vector", None)

    return JSONResponse(
        content={
            "resume": {
                "skills": resume_data.get("skills"),
                "experience": resume_data.get("experience"),
                "education": resume_data.get("education"),
            },
            "top_jobs": ranked,
        }
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_fastapi:app", host="127.0.0.1", port=8000, reload=True)
