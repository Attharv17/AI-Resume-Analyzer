"""
ranking_service.py
------------------
Unified ranking service for both Flask and FastAPI.
Ranks jobs against a resume using semantic embeddings and experience matching.
"""

from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
import functools

from services.ats_engine import compute_semantic_match
from utils.skill_extractor import extract_skills

# ---------------------------------------------------------------------------
# Feature logic merged from old features.py
# ---------------------------------------------------------------------------
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


def generate_recommendation_reason(score: int, missing_skills: List[str]) -> str:
    """Generate a qualitative reason for the recommendation."""
    if score >= 80:
        if not missing_skills:
            return "Highly recommended: Perfect alignment with the required skills."
        elif len(missing_skills) <= 2:
            return f"Highly recommended: Strong semantic match, though missing {', '.join(missing_skills)}."
        else:
            return "Highly recommended: Very strong semantic match overall."
    elif score >= 65:
        if len(missing_skills) > 0:
            return f"Good match: Solid alignment, but missing some skills like {missing_skills[0]}."
        else:
            return "Good match: Aligns reasonably well with your profile."
    elif score >= 50:
        if missing_skills:
            return f"Partial match: Missing key requirements such as {', '.join(missing_skills[:2])}."
        else:
            return "Partial match: Moderate semantic alignment."
    else:
        return "Low match: This role may require significantly different experience or skills."

# ---------------------------------------------------------------------------
# FastAPI Ranking Logic (was ranking_engine.py)
# ---------------------------------------------------------------------------
def rank_jobs(
    resume: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    
    resume_skills = list(resume.get("skills") or [])
    resume_text_parts = [
        str(resume.get("education") or ""),
        " ".join(resume_skills),
        str(resume.get("raw_text") or ""),
    ]
    resume_blob = "\n".join(p for p in resume_text_parts if p)
    resume_years = int(resume.get("experience") or 0)

    for job in jobs:
        job_skills = list(job.get("skills_required") or [])
        job_years = int(job.get("experience_required") or 0)
        desc = str(job.get("description") or "")
        job_blob = "\n".join([str(job.get("title") or ""), desc, " ".join(job_skills)])

        match_result = compute_semantic_match(
            resume_text=resume_blob,
            job_text=job_blob,
            resume_skills=resume_skills,
            job_skills=job_skills,
        )

        # Apply experience as a gentle scaling multiplier (0.80–1.00×) so that
        # under-qualified candidates lose points without dominating the score.
        # We deliberately do NOT use the RandomForestRegressor here — its
        # tree-averaging behaviour compresses all predictions toward the mean.
        em = experience_match_score(resume_years, job_years)
        experience_scale = 0.80 + 0.20 * em  # range: 0.80 (0 yrs) → 1.00 (meets req)

        raw_engine_score = match_result["score"]
        final_score = int(round(min(max(raw_engine_score * experience_scale, 0), 100)))

        missing = match_result["missing_skills"]
        reason = generate_recommendation_reason(final_score, missing)

        ranked.append(
            {
                "title": job.get("title", ""),
                "skills_required": job_skills,
                "experience_required": job_years,
                "description": desc,
                "score": final_score,
                "confidence": match_result.get("confidence", 0.0),
                "matched_skills": match_result.get("matched_skills", []),
                "missing_skills": missing,
                "recommendation_reason": reason,
                "scoring_method": match_result.get("scoring_method", "semantic"),
            }
        )

    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Flask Ranking Logic (was ranker.py)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _load_and_parse_jobs(jobs_csv_path: str) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(jobs_csv_path)
    except Exception as e:
        raise RuntimeError(f"Could not read jobs dataset: {e}")

    required_cols = {"title", "description", "skills", "location"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"jobs.csv missing required columns. Needed: {required_cols}")

    parsed_jobs = []
    for _, row in df.iterrows():
        title = str(row["title"])
        description = str(row["description"])
        csv_skills = str(row["skills"])
        full_job_text = description + " " + csv_skills
        job_skills = extract_skills(full_job_text)
        
        parsed_jobs.append({
            "title": title,
            "full_job_text": full_job_text,
            "job_skills": job_skills
        })
    return parsed_jobs


def rank_jobs_for_resume(
    resume_text: str,
    resume_skills: List[str],
    jobs_csv_path: str
) -> Dict[str, Any]:
    """
    Ranks multiple jobs for a given resume text and skills from CSV.
    Used by Flask `/rank_jobs` route.
    """
    parsed_jobs = _load_and_parse_jobs(jobs_csv_path)
    ranked_jobs = []

    for job in parsed_jobs:
        match_result = compute_semantic_match(
            resume_text=resume_text,
            job_text=job["full_job_text"],
            resume_skills=resume_skills,
            job_skills=job["job_skills"],
        )

        # Use engine score directly — no ML model override.
        # CSV jobs may not have experience_required; default experience_scale=1.0.
        score = min(max(match_result["score"], 0), 100)
        missing = match_result["missing_skills"]
        reason = generate_recommendation_reason(score, missing)

        ranked_jobs.append({
            "title": job["title"],
            "score": score,
            "confidence": match_result.get("confidence", 0.0),
            "matched_skills": match_result.get("matched_skills", []),
            "missing_skills": missing,
            "recommendation_reason": reason,
            "scoring_method": match_result.get("scoring_method", "semantic"),
        })

    ranked_jobs.sort(key=lambda x: x["score"], reverse=True)
    return {"top_jobs": ranked_jobs[:5]}
