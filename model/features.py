"""
features.py (Phase 4)
---------------------
Compute feature vector from resume JSON + job JSON:
  - skill_match_score (0–1)
  - experience_match_score (0–1)
  - text_similarity_score (0–1) via cosine similarity (TF-IDF)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model.job_requirements import extract_job_requirements


def skill_match_score(resume_skills: List[str], job_skills: List[str]) -> float:
    """Fraction of required job skills present on the resume: |R ∩ J| / |J| (0–1)."""
    if not job_skills:
        return 1.0
    rs = {s.lower().strip() for s in resume_skills if str(s).strip()}
    js = [s.lower().strip() for s in job_skills if str(s).strip()]
    if not js:
        return 1.0
    js_set = set(js)
    inter = len(rs & js_set)
    return float(inter) / float(len(js_set))


def experience_match_score(resume_years: int, required_years: int) -> float:
    """
    1.0 if resume experience meets/exceeds required years.
    Smooth decay if below requirement (0–1).
    """
    if required_years <= 0:
        return 1.0
    r = max(0, int(resume_years))
    req = max(0, int(required_years))
    if r >= req:
        return 1.0
    return max(0.0, min(1.0, r / float(req)))



# Reusable vectorizer — constructed once, re-fit on each pair of documents.
# Avoids repeated object creation and stop-word list compilation overhead.
_TFIDF_VECTORIZER = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
)

def text_similarity_score(resume_blob: str, job_blob: str) -> float:
    """Cosine similarity of TF-IDF vectors (0–1).

    Uses a module-level vectorizer instance to avoid re-instantiating the
    stop-word list and vocabulary on every call.
    """
    a = (resume_blob or "").strip()
    b = (job_blob or "").strip()
    if not a or not b:
        return 0.0
    try:
        m = _TFIDF_VECTORIZER.fit_transform([a, b])
        sim = cosine_similarity(m[0:1], m[1:2])[0][0]
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.0



def build_feature_vector(
    resume: Dict[str, Any],
    job: Dict[str, Any],
) -> Dict[str, float]:
    """
    Input shapes:
      resume: { "skills": [...], "experience": int, optional raw "text" }
      job: { "skills_required": [...], "experience_required": int, "description": str, optional "title" }
    """
    resume_skills = list(resume.get("skills") or [])
    resume_years = int(resume.get("experience") or 0)
    resume_text_parts = [
        str(resume.get("education") or ""),
        " ".join(resume_skills),
        str(resume.get("raw_text") or ""),
    ]
    resume_blob = "\n".join(p for p in resume_text_parts if p)

    job_skills = list(job.get("skills_required") or [])
    job_years = int(job.get("experience_required") or 0)
    desc = str(job.get("description") or "")
    if not job_skills and desc:
        extracted = extract_job_requirements(desc)
        job_skills = extracted["skills_required"]
        if not job_years:
            job_years = extracted["experience_required"]
    job_blob = "\n".join(
        [str(job.get("title") or ""), desc, " ".join(job_skills)]
    )

    sm = skill_match_score(resume_skills, job_skills)
    em = experience_match_score(resume_years, job_years)
    tm = text_similarity_score(resume_blob, job_blob)

    return {
        "skill_match": sm,
        "experience_match": em,
        "similarity_score": tm,
    }


def feature_tuple(fv: Dict[str, float]) -> Tuple[float, float, float]:
    return (fv["skill_match"], fv["experience_match"], fv["similarity_score"])
