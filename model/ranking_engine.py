"""
ranking_engine.py (Phase 7 - Semantic Ranking)
----------------------------------------------
Rank all jobs for a resume using semantic embeddings and experience matching.
Returns: top N jobs with normalized match %, missing skills, and a recommendation reason.
"""

from __future__ import annotations

from typing import Any, Dict, List

from model.similarity_engine import compute_semantic_match
from model.features import experience_match_score


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

        # 1. Base semantic score
        match_result = compute_semantic_match(
            resume_text=resume_blob,
            job_text=job_blob,
            resume_skills=resume_skills,
            job_skills=job_skills,
        )
        
        # 2. Experience weighting
        # We weigh semantic score 85% and experience match 15% to maintain realistic 0-100 scales
        em = experience_match_score(resume_years, job_years)
        semantic_score = match_result["score"]
        final_score = int(round((semantic_score * 0.85) + (em * 100 * 0.15)))
        
        final_score = min(max(final_score, 0), 100)
        missing = match_result["missing_skills"]
        
        reason = generate_recommendation_reason(final_score, missing)

        ranked.append(
            {
                "title": job.get("title", ""),
                "skills_required": job_skills,
                "experience_required": job_years,
                "description": desc,
                "score": final_score,
                "missing_skills": missing,
                "recommendation_reason": reason,
            }
        )

    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[:top_n]
