"""
inference_pipeline.py  (formerly matcher.py)
---------------------------------------------
Computes the match between a resume and a job description.

Scoring strategy
----------------
Delegates entirely to ats_engine.compute_semantic_match which implements:
  - Full-text sigmoid-stretched cosine similarity   (30 % weight)
  - Exact skill set-intersection overlap            (45 % weight)
  - Semantic skill coverage ratio                   (25 % weight)
  - Calibrated penalty/boost adjustments
  - Detailed debug logging of every component

The RandomForestRegressor score override that previously lived here has been
removed.  The RF model was trained on synthetic data with experience_match
hardcoded to 1.0 and its tree-averaging behaviour compressed all predictions
into a narrow 70-80 % band.  Ranking callers (ranking_service.py) apply
experience scaling as a multiplicative factor after the engine score.

Response shape (unchanged — backward compatible)
-------------------------------------------------
    {
        "score":          int   (0-100),
        "confidence":     float (0.0-1.0),
        "matched_skills": List[str],
        "missing_skills": List[str],
        "scoring_method": str,          # "semantic" | "tfidf_fallback"
        "score_breakdown": dict,        # component-level debug info
    }
"""

from typing import Dict, Any, List

from services.ats_engine import compute_semantic_match


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_match(
    resume_text: str,
    job_text: str,
    resume_skills: List[str],
    job_skills: List[str],
) -> Dict[str, Any]:
    """
    Compute the overall match between a resume and a job description.

    Delegates to ats_engine.compute_semantic_match which uses sentence-
    transformer embeddings for full-text similarity and semantic skill
    matching, with sigmoid stretching, exact overlap scoring, and
    penalty/boost calibration.  Falls back to TF-IDF automatically if
    embeddings are unavailable.

    Args:
        resume_text   (str)       : Raw extracted text from the resume PDF.
        job_text      (str)       : Raw job description text.
        resume_skills (List[str]) : Skills extracted from the resume.
        job_skills    (List[str]) : Skills extracted from the job description.

    Returns:
        Dict with keys:
            "score"          – Calibrated ATS score 0-100 (int).
            "confidence"     – Mean skill-level match confidence (float 0-1).
            "matched_skills" – Skills matched between resume & JD.
            "missing_skills" – Job skills not found in the resume.
            "scoring_method" – "semantic" or "tfidf_fallback".
            "score_breakdown"– Per-component debug breakdown (dict).
    """
    return compute_semantic_match(
        resume_text=resume_text,
        job_text=job_text,
        resume_skills=resume_skills,
        job_skills=job_skills,
    )
