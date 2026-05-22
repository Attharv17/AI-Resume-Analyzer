"""
extractor.py
------------
Backward-compatible shim that delegates to the enhanced skill_extractor module.

All existing callers (app.py, comparator.py, ranker.py, resume_structured.py,
job_requirements.py, features.py) continue to call::

    from model.extractor import extract_skills

…and receive the improved 250+ vocabulary output without any import changes.

To access section-aware or spaCy-augmented extraction, import directly from
model.skill_extractor instead.
"""

from __future__ import annotations

from typing import List

# Re-export the enhanced implementation transparently
from model.skill_extractor import (   # noqa: F401  (re-exported)
    SKILL_ALIASES,
    detect_sections,
    extract_skills,
    extract_skills_detailed,
    normalize_skill,
)

# Keep SKILL_LIBRARY as an alias so any code that imports it directly still works
from model.skill_extractor import _CANONICAL as _vocab

SKILL_LIBRARY: List[str] = sorted(_vocab.keys())
