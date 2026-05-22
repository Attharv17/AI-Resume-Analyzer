"""
ranker.py
---------
Handles matching a single resume against a dataset of jobs.
Loads the dataset via pandas, applies the match logic from matcher.py,
and returns the top ranked jobs.
"""

import pandas as pd
from typing import Dict, Any, List

from model.skill_extractor import extract_skills
from model.matcher import compute_match

import functools

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
    Ranks multiple jobs for a given resume text and skills.
    
    Args:
        resume_text (str): Extracted text of the resume.
        resume_skills (List[str]): Skills extracted from the resume.
        jobs_csv_path (str): Path to the jobs.csv dataset.
        
    Returns:
        Dict returning top 5 jobs in the required structure.
    """
    parsed_jobs = _load_and_parse_jobs(jobs_csv_path)

    ranked_jobs = []

    for job in parsed_jobs:
        match_result = compute_match(
            resume_text=resume_text,
            job_text=job["full_job_text"],
            resume_skills=resume_skills,
            job_skills=job["job_skills"],
        )
        
        score = match_result["score"]
        missing = match_result["missing_skills"]
        from model.ranking_engine import generate_recommendation_reason
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

    # Sort descending by score
    ranked_jobs.sort(key=lambda x: x["score"], reverse=True)

    # Return top 5
    return {
        "top_jobs": ranked_jobs[:5]
    }
