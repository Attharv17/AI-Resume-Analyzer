"""
resume_structured.py
--------------------
Parse a PDF resume into structured fields:
  - skills (list[str])
  - experience (years as int)
  - education (single summary string)

Uses:
  - PyMuPDF text extraction (via model.parser)
  - Keyword skill matching (model.extractor)
  - Optional spaCy (en_core_web_sm) for light cleanup / education line hints
  - Regex heuristics for years-of-experience and education sections
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from model.extractor import extract_skills
from model.parser import extract_text_from_pdf

# ---------------------------------------------------------------------------
# Optional spaCy
# ---------------------------------------------------------------------------
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is False:
        return None
    if _NLP is not None:
        return _NLP
    try:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = False
    return _NLP if _NLP is not False else None


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _split_sections(text: str) -> Dict[str, str]:
    """
    Split resume into rough sections by common headings (case-insensitive).
    Unmatched content goes under '_rest'.
    """
    headings = [
        "education",
        "academic",
        "experience",
        "work experience",
        "employment",
        "professional experience",
        "skills",
        "technical skills",
        "projects",
        "summary",
        "objective",
    ]
    pattern = r"(?mi)^\s*(" + "|".join(re.escape(h) for h in sorted(headings, key=len, reverse=True)) + r")\s*:?\s*$"
    parts = re.split(pattern, text)
    sections: Dict[str, str] = {}
    if not parts:
        return {"_rest": text}
    if parts[0].strip():
        sections["_rest"] = parts[0].strip()
    i = 1
    while i + 1 < len(parts):
        name = parts[i].strip().lower()
        body = parts[i + 1].strip()
        sections[name] = (sections.get(name, "") + "\n" + body).strip()
        i += 2
    return sections


# ---------------------------------------------------------------------------
# Experience (years)
# ---------------------------------------------------------------------------

_YEAR_RANGE_RE = re.compile(
    r"(?P<a>\b(19|20)\d{2}\b)\s*[-–—]\s*(?P<b>\b(19|20)\d{2}\b|\bpresent\b)",
    re.I,
)
_EXPLICIT_YEARS_RE = re.compile(
    r"(?P<n>\d{1,2})\+?\s*(?:\+?\s*)?(?:years?|yrs?\.?)\b(?:\s+of)?(?:\s+experience)?",
    re.I,
)


def _years_from_range(a: str, b: str, now_year: int) -> int:
    try:
        y1 = int(re.search(r"\d{4}", a).group())
    except Exception:
        return 0
    if re.search(r"present", b, re.I):
        y2 = now_year
    else:
        try:
            y2 = int(re.search(r"\d{4}", b).group())
        except Exception:
            return 0
    if y2 < y1:
        y1, y2 = y2, y1
    return max(0, y2 - y1)


def extract_experience_years(text: str) -> int:
    """
    Heuristic total years of experience:
    - Max of explicit 'N years' phrases
    - Max span of 4-digit year ranges (incl. Present)
    Returns a non-negative int (capped at 45).
    """
    now_year = datetime.now().year
    text_n = _normalize_whitespace(text)
    scores: List[int] = []

    for m in _EXPLICIT_YEARS_RE.finditer(text_n):
        try:
            n = int(m.group("n"))
            scores.append(min(n, 45))
        except Exception:
            continue

    for m in _YEAR_RANGE_RE.finditer(text_n):
        span = _years_from_range(m.group("a"), m.group("b"), now_year)
        if span:
            scores.append(min(span, 45))

    if not scores:
        return 0
    return int(min(max(scores), 45))


# ---------------------------------------------------------------------------
# Education (string)
# ---------------------------------------------------------------------------

_DEGREE_HINT = re.compile(
    r"\b("
    r"ph\.?d\.?|doctor(?:ate)?|m\.?s\.?|m\.?a\.?|mba|m\.?eng\.?|"
    r"b\.?s\.?|b\.?a\.?|bachelor|master|associate|diploma|certificate"
    r")\b",
    re.I,
)


def extract_education_string(text: str, sections: Optional[Dict[str, str]] = None) -> str:
    """
    Prefer content under Education / Academic sections; else first degree-like block.
    """
    sections = sections or _split_sections(text)
    for key in ("education", "academic"):
        if key in sections and sections[key].strip():
            block = sections[key].strip()
            # First 1-3 non-empty lines often hold degree + school
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            snippet = " ".join(lines[:3])
            return snippet[:500]

    nlp = _get_nlp()
    if nlp:
        doc = nlp(text[:100000])
        for sent in doc.sents:
            s = sent.text.strip()
            if _DEGREE_HINT.search(s) and len(s) > 15:
                return s[:500]

    lines = [ln.strip() for ln in _normalize_whitespace(text).splitlines() if ln.strip()]
    for ln in lines:
        if _DEGREE_HINT.search(ln):
            return ln[:500]

    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_resume_pdf(pdf_path: str | Path) -> Dict[str, Any]:
    """
    Parse a PDF resume file into structured fields.

    Returns:
        {
          "skills": [...],
          "experience": <int>,
          "education": "<string>",
          "meta": { "source_pdf": "...", "text_length": N }
        }
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    raw_text = extract_text_from_pdf(str(path))
    text = _normalize_whitespace(raw_text)
    sections = _split_sections(text)

    skills = extract_skills(text)
    years = extract_experience_years(text)
    education = extract_education_string(text, sections)

    return {
        "skills": skills,
        "experience": years,
        "education": education,
        # Used by matching / TF-IDF; omit when serializing to clients if desired
        "raw_text": text[:50000],
        "meta": {
            "source_pdf": str(path.resolve()),
            "text_length": len(text),
        },
    }


def parse_resume_to_json_file(
    pdf_path: str | Path,
    out_path: str | Path = "resume_data.json",
    indent: int = 2,
) -> Dict[str, Any]:
    """Parse resume and write structured JSON to disk."""
    data = parse_resume_pdf(pdf_path)
    out = Path(out_path)
    out.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    return data


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extract structured data from a PDF resume.")
    p.add_argument("pdf", help="Path to PDF resume")
    p.add_argument(
        "-o",
        "--output",
        default="resume_data.json",
        help="Output JSON path (default: resume_data.json)",
    )
    args = p.parse_args()
    parse_resume_to_json_file(args.pdf, args.output)
    print(f"Wrote {args.output}")
