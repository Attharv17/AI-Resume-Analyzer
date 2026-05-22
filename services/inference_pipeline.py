"""
matcher.py
----------
Computes the match between a resume and a job description.

Scoring strategy (updated)
---------------------------
Primary  : Semantic similarity via sentence-transformers/all-MiniLM-L6-v2
           (model.similarity_engine.compute_semantic_match)
           – Full-text cosine similarity of dense embeddings (70 % weight)
           – Semantic skill-coverage ratio (30 % weight)

Fallback : TF-IDF cosine similarity (original approach, used automatically
           if sentence-transformers is not installed or fails to load).

Response shape
--------------
    {
        "score":          int   (0-100),   # blended semantic score
        "confidence":     float (0.0-1.0), # mean skill-match confidence
        "matched_skills": List[str],       # semantically matched skills
        "missing_skills": List[str],       # job skills absent from resume
        "scoring_method": str,             # "semantic" | "tfidf_fallback"
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

    Delegates to :func:`model.similarity_engine.compute_semantic_match` which
    uses sentence-transformer embeddings for both full-text similarity and
    semantic skill matching. Falls back to TF-IDF automatically if embeddings
    are unavailable.

    Args:
        resume_text   (str)       : Raw extracted text from the resume PDF.
        job_text      (str)       : Raw job description text (from the form).
        resume_skills (List[str]) : Skills extracted from the resume.
        job_skills    (List[str]) : Skills extracted from the job description.

    Returns:
        Dict with keys:
            "score"          – Semantic similarity scaled to 0–100 (int).
            "confidence"     – Mean skill-level match confidence (float 0–1).
            "matched_skills" – Skills semantically matched between resume & JD.
            "missing_skills" – Job skills not found in the resume.
            "scoring_method" – "semantic" or "tfidf_fallback".
    """
    return compute_semantic_match(
        resume_text=resume_text,
        job_text=job_text,
        resume_skills=resume_skills,
        job_skills=job_skills,
    )
