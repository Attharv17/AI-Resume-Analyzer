"""
ranking_engine.py (Phase 7)
---------------------------
Rank all jobs for a resume:
  - compute score per job
  - sort descending
  - return top N (default 10)
"""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.ensemble import RandomForestRegressor

from model.features import build_feature_vector
from model.hybrid_scorer import hybrid_from_feature_dict
from model.skill_gap import missing_skills


def rank_jobs(
    resume: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    model: RandomForestRegressor,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    resume_skills = list(resume.get("skills") or [])

    for job in jobs:
        fv = build_feature_vector(resume, job)
        final, dbg = hybrid_from_feature_dict(fv, model, scale_0_100=True)
        job_skills = list(job.get("skills_required") or [])
        missing = missing_skills(resume_skills, job_skills)

        ranked.append(
            {
                "title": job.get("title", ""),
                "skills_required": job_skills,
                "experience_required": int(job.get("experience_required") or 0),
                "description": job.get("description", ""),
                "score": round(float(final), 3),
                "feature_vector": fv,
                "debug": dbg,
                "missing_skills": missing,
            }
        )

    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[:top_n]
