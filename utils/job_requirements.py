"""
job_requirements.py (Phase 3)
-----------------------------
Extract structured requirements from raw job descriptions:
  - skills_required (list[str])
  - experience_required (int, years; 0 if not specified)
"""

from __future__ import annotations

import re
from typing import Any, Dict

from utils.skill_extractor import extract_skills

# Prefer explicit job-posting phrasing first
_MIN_YEARS_RE = re.compile(
    r"(?:minimum|at least|more than|over|upto|up to|around)?\s*"
    r"(?P<n>\d{1,2})\+?\s*(?:\+?\s*)?(?:years?|yrs?\.?)\b",
    re.I,
)


def extract_experience_required(text: str) -> int:
    """
    Minimum years required from phrases like '3+ years', 'at least 5 years'.
    Returns 0 if nothing plausible is found (capped at 25).
    """
    if not text or not text.strip():
        return 0
    t = text.lower()
    candidates: list[int] = []

    for m in _MIN_YEARS_RE.finditer(t):
        try:
            n = int(m.group("n"))
            candidates.append(min(max(n, 0), 25))
        except Exception:
            continue

    # Heuristic: take minimum of stated requirements (job asks for "3-5 years" -> 3)
    if candidates:
        return min(candidates)
    return 0


def extract_skills_required(text: str) -> list[str]:
    """Keyword-based skill extraction from job text (same vocabulary as resume)."""
    return extract_skills(text)


def extract_job_requirements(description: str) -> Dict[str, Any]:
    """
    Clean structured output for one job description.

    Returns:
        {"skills_required": [...], "experience_required": int}
    """
    return {
        "skills_required": extract_skills_required(description),
        "experience_required": extract_experience_required(description),
    }
