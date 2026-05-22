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
    try:
        df = pd.read_csv(jobs_csv_path)
    except Exception as e:
        raise RuntimeError(f"Could not read jobs dataset: {e}")

    # Check required columns
    required_cols = {"title", "description", "skills", "location"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"jobs.csv missing required columns. Needed: {required_cols}")

    ranked_jobs = []

    for _, row in df.iterrows():
        title = str(row["title"])
        description = str(row["description"])
        csv_skills = str(row["skills"])
        
        # Combine description and csv skills for thorough skill extraction and TF-IDF
        full_job_text = description + " " + csv_skills
        
        # Extract skills for the job
        job_skills = extract_skills(full_job_text)
        
        # Compute match score against resume
        match_result = compute_match(
            resume_text=resume_text,
            job_text=full_job_text,
            resume_skills=resume_skills,
            job_skills=job_skills,
        )
        
        ranked_jobs.append({
            "title": title,
            "score": match_result["score"],
            "missing_skills": match_result["missing_skills"]
        })

    # Sort descending by score
    ranked_jobs.sort(key=lambda x: x["score"], reverse=True)

    # Return top 5
    return {
        "top_jobs": ranked_jobs[:5]
    }
