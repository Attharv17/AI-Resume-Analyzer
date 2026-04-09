"""
extractor.py
------------
Responsible for extracting skills from a block of text by matching
against a predefined skill keyword list (case-insensitive).
"""

import re
from typing import List

# ---------------------------------------------------------------------------
# Predefined skill vocabulary
# Add / remove skills here to grow the matching capability.
# ---------------------------------------------------------------------------
SKILL_LIBRARY: List[str] = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#",
    "r", "scala", "kotlin", "swift", "go", "rust", "ruby", "php",
    "bash", "shell scripting", "perl",

    # Web / frontend
    "html", "css", "react", "angular", "vue", "next.js", "node.js",
    "express", "django", "flask", "fastapi", "spring boot", "rest api",
    "graphql", "jquery",

    # Data & ML
    "machine learning", "deep learning", "natural language processing",
    "nlp", "computer vision", "data analysis", "data science",
    "data engineering", "feature engineering", "model deployment",
    "statistics", "statistical analysis", "time series",

    # ML frameworks & tools
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost",
    "lightgbm", "hugging face", "transformers", "opencv",

    # Data tools
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau",
    "power bi", "excel", "jupyter",

    # Databases & query
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "redis",
    "cassandra", "elasticsearch", "oracle", "nosql",

    # Big Data
    "spark", "hadoop", "kafka", "hive", "airflow", "dbt",

    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "ansible", "jenkins", "ci/cd", "git", "github",
    "linux", "unix",

    # Soft / general
    "agile", "scrum", "jira", "communication", "leadership",
    "problem solving", "teamwork",
]


def extract_skills(text: str) -> List[str]:
    """
    Scan *text* and return every skill from SKILL_LIBRARY that appears in it.

    Matching rules:
    - Case-insensitive.
    - Whole-word / whole-phrase matching (surrounded by word boundaries or
      non-alphanumeric characters) to avoid false positives.

    Args:
        text (str): Raw text (resume or job description).

    Returns:
        List[str]: Sorted list of unique matched skills (lowercase).
    """
    text_lower = text.lower()
    found: set = set()

    for skill in SKILL_LIBRARY:
        # Build a regex that matches the skill as a whole phrase.
        # \b works for single words; for multi-word skills we use lookahead/
        # lookbehind that assert a non-alphanumeric boundary.
        if " " in skill or "." in skill or "+" in skill or "#" in skill:
            # Escape special regex chars, then wrap with boundary assertions.
            pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        else:
            pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text_lower):
            found.add(skill)

    return sorted(found)
