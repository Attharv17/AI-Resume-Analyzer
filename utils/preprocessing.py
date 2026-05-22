"""
preprocessing.py
----------------
Text-cleaning pipeline for resume content.

Pipeline stages (applied in order)
-----------------------------------
1.  Unicode / ligature normalisation        – handled upstream in parser.py
2.  Page-number line removal                – strips lone digit lines
3.  URL & email stripping                   – removes hyperlinks (not needed for scoring)
4.  Noise character removal                 – strips junk symbols (■, ▪, ●, bars, etc.)
5.  Repeated-punctuation collapse           – "------" → "-"
6.  Whitespace normalisation                – collapses internal spaces, strips blank lines
7.  Short-line noise filter                 – removes lines with fewer than 2 characters
8.  Section heading normalisation           – strips trailing colons, normalises casing

Public API
----------
    clean_text(raw: str) -> str
    clean_line(line: str) -> str
    remove_noise_lines(lines: list[str]) -> list[str]
"""

from __future__ import annotations

import re
from typing import List


# ---------------------------------------------------------------------------
# Compiled regex patterns (compiled once at import time for speed)
# ---------------------------------------------------------------------------

# Page numbers — lone number, possibly prefixed with "Page" or "p."
_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?p?\.?\s*\d{1,4}\s*$", re.I)

# Bare URLs  (http/https/ftp/www...)
_URL_RE = re.compile(
    r"https?://[^\s]+|www\.[^\s]+|ftp://[^\s]+",
    re.I,
)

# Email addresses
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Phone numbers  (+91-9876543210, (123) 456-7890, etc.) – kept: useful context
# Intentionally NOT removed so the text still contains context clues.

# Noise / separator characters — lines that are ONLY these
_ONLY_SYMBOLS_RE = re.compile(r"^[\s\-=_|•·◆▪■◇▶▸►▲▼★☆✓✔✗✘\*\.,:;\/\\]+$")

# Repeated punctuation runs (--- or === or ___) that add no content
_REPEATED_PUNCT_RE = re.compile(r"([-=_|]{3,})")

# Collapse internal whitespace
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Lone special chars that show up as bullets in PDFs but aren't real unicode bullets
_STRAY_GLYPH_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Section heading normalisation — "SKILLS:" → "SKILLS", "  Education  " → "Education"
_HEADING_TRAILING_COLON_RE = re.compile(r":\s*$")


# ---------------------------------------------------------------------------
# Individual cleaning functions
# ---------------------------------------------------------------------------

def _strip_urls(text: str) -> str:
    """Remove bare URLs from text (LinkedIn, portfolio links etc. add noise)."""
    return _URL_RE.sub("", text)


def _strip_emails(text: str) -> str:
    """Remove email addresses (not useful for skill matching)."""
    return _EMAIL_RE.sub("", text)


def _strip_stray_glyphs(text: str) -> str:
    """Remove non-printable control characters."""
    return _STRAY_GLYPH_RE.sub("", text)


def _collapse_repeated_punct(line: str) -> str:
    """Collapse runs of ---, ===, ___ into a single dash."""
    return _REPEATED_PUNCT_RE.sub("-", line)


def _normalise_whitespace_in_line(line: str) -> str:
    """Collapse multiple spaces/tabs within a line to one space."""
    return _MULTI_SPACE_RE.sub(" ", line).strip()


def _is_noise_line(line: str) -> bool:
    """
    Return True if this line carries no useful textual content.
    Criteria:
      - Empty after stripping
      - Looks like a page number
      - Made entirely of separator / symbol characters
      - Shorter than 2 characters after cleaning
    """
    stripped = line.strip()
    if not stripped:
        return True
    if _PAGE_NUM_RE.match(stripped):
        return True
    if _ONLY_SYMBOLS_RE.match(stripped):
        return True
    if len(stripped) < 2:
        return True
    return False


def clean_line(line: str) -> str:
    """
    Apply all single-line transformations.

    Args:
        line (str): A single text line from the PDF.

    Returns:
        str: Cleaned line.
    """
    line = _strip_stray_glyphs(line)
    line = _strip_urls(line)
    line = _strip_emails(line)
    line = _collapse_repeated_punct(line)
    line = _normalise_whitespace_in_line(line)
    # Remove trailing colon from headings (e.g. "SKILLS:" → "SKILLS")
    line = _HEADING_TRAILING_COLON_RE.sub("", line)
    return line.strip()


def remove_noise_lines(lines: List[str]) -> List[str]:
    """
    Filter a list of lines, dropping those identified as noise.

    Args:
        lines (List[str]): Pre-cleaned text lines.

    Returns:
        List[str]: Lines containing actual textual content.
    """
    return [ln for ln in lines if not _is_noise_line(ln)]


# ---------------------------------------------------------------------------
# Duplicate-line deduplication within a single resume
# ---------------------------------------------------------------------------

def _deduplicate_lines(lines: List[str]) -> List[str]:
    """
    Remove exact duplicate lines that may arise from two-column reflow artefacts.
    Preserves first occurrence and original order.
    """
    seen: set[str] = set()
    result: List[str] = []
    for ln in lines:
        key = ln.lower().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(ln)
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """
    Run the complete preprocessing pipeline on raw PDF-extracted text.

    Steps:
        1. Split into lines
        2. Clean each line (strip glyphs, URLs, emails, collapse punctuation)
        3. Filter noise lines
        4. Deduplicate lines
        5. Re-join with newlines

    Args:
        raw (str): Raw text from PDF extraction.

    Returns:
        str: Cleaned text suitable for section detection and skill extraction.
    """
    if not raw or not raw.strip():
        return ""

    lines = raw.splitlines()
    lines = [clean_line(ln) for ln in lines]
    lines = remove_noise_lines(lines)
    lines = _deduplicate_lines(lines)

    return "\n".join(lines).strip()
