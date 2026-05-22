"""
skill_gap.py (Phase 10 / shared)
-------------------------------
Compare resume skills vs job skills; return missing skills for gap analysis.
"""

from __future__ import annotations

from typing import List, Set


def normalize_skill(s: str) -> str:
    return str(s).strip().lower()


def missing_skills(resume_skills: List[str], job_skills: List[str]) -> List[str]:
    """Skills required by the job that are not present on the resume (normalized)."""
    rs: Set[str] = {normalize_skill(s) for s in resume_skills if str(s).strip()}
    missing: List[str] = []
    for s in job_skills:
        ns = normalize_skill(s)
        if not ns:
            continue
        if ns not in rs:
            missing.append(ns)
    return sorted(set(missing))
