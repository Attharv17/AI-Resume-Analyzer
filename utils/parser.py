"""
parser.py
---------
Enhanced PDF text extractor for resume parsing.

Strategy
--------
1. Primary   – PyMuPDF "blocks" mode with spatial sorting so multi-column
               layouts are read left-to-right, top-to-bottom instead of
               concatenating columns vertically.
2. Fallback  – pdfplumber (if installed) for PDFs where block extraction
               returns empty or garbled text.
3. Post-pass – preprocessing.clean_text() removes noise before returning.

Public API (unchanged – drop-in replacement)
--------------------------------------------
    extract_text_from_pdf(file_path: str) -> str
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Ligature / unicode normalisation map
# ---------------------------------------------------------------------------
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u2019": "'",   # right single quote
    "\u2018": "'",   # left single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2022": "-",   # bullet •
    "\u2023": "-",   # triangular bullet
    "\u25cf": "-",   # black circle bullet
    "\u25ba": "-",   # black right-pointing pointer
    "\u2012": "-",   # figure dash
}

_LIGATURE_RE = re.compile("|".join(re.escape(k) for k in _LIGATURE_MAP))


def _fix_ligatures(text: str) -> str:
    """Replace common ligature / fancy unicode characters with ASCII equivalents."""
    text = _LIGATURE_RE.sub(lambda m: _LIGATURE_MAP[m.group()], text)
    # NFKD decompose remaining composed chars, keep printable ASCII + basic latin
    return unicodedata.normalize("NFKD", text)


# ---------------------------------------------------------------------------
# Block-level spatial sort helpers
# ---------------------------------------------------------------------------

_ROW_BUCKET = 15  # px tolerance — blocks within this y-range share a "row"


def _row_key(y: float) -> int:
    """Snap y-coordinate to nearest row bucket for stable multi-column sorting."""
    return int(y // _ROW_BUCKET)


def _extract_page_blocks(page) -> str:
    """
    Extract text from a single PyMuPDF page using block-level spatial sorting.

    Blocks are sorted first by their bucketed top-y coordinate (top-to-bottom),
    then by x (left-to-right within the same visual row).  This correctly handles
    two-column and three-column resume layouts.
    """
    blocks = page.get_text("blocks")   # list of (x0, y0, x1, y1, text, block_no, block_type)
    # Filter to text blocks only (block_type == 0) and non-empty
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    # Sort: primary = row bucket (y0), secondary = x0
    text_blocks.sort(key=lambda b: (_row_key(b[1]), b[0]))
    return "\n".join(b[4].strip() for b in text_blocks)


def _deduplicate_headers_footers(pages: List[str]) -> List[str]:
    """
    Remove text lines that appear on every page (likely headers/footers).
    Only applied when there are 2+ pages.
    """
    if len(pages) < 2:
        return pages

    # Build a frequency map of exact lines
    from collections import Counter
    all_lines: list[str] = []
    for page_text in pages:
        all_lines.extend(ln.strip() for ln in page_text.splitlines() if ln.strip())

    freq = Counter(all_lines)
    repeated: set[str] = {ln for ln, count in freq.items() if count >= len(pages)}

    if not repeated:
        return pages

    cleaned = []
    for page_text in pages:
        lines = [ln for ln in page_text.splitlines() if ln.strip() not in repeated]
        cleaned.append("\n".join(lines))
    return cleaned


# ---------------------------------------------------------------------------
# PyMuPDF extractor (primary)
# ---------------------------------------------------------------------------

def _extract_with_pymupdf(file_path: str) -> str:
    """
    Extract text using PyMuPDF (fitz) with block-level spatial sorting.
    Returns empty string if fitz is not installed or extraction fails.
    """
    try:
        import fitz  # noqa: PLC0415
    except ImportError:
        return ""

    try:
        pages: List[str] = []
        with fitz.open(file_path) as doc:
            for page in doc:
                page_text = _extract_page_blocks(page)
                if not page_text.strip():
                    # Fallback for this page to plain text mode
                    page_text = page.get_text("text")
                pages.append(page_text)

        pages = _deduplicate_headers_footers(pages)
        return "\n\n".join(p for p in pages if p.strip()).strip()

    except Exception:
        return ""


# ---------------------------------------------------------------------------
# pdfplumber extractor (fallback)
# ---------------------------------------------------------------------------

def _extract_with_pdfplumber(file_path: str) -> str:
    """
    Fallback extractor using pdfplumber.
    Returns empty string if pdfplumber is not installed or extraction fails.
    """
    try:
        import pdfplumber  # noqa: PLC0415
    except ImportError:
        return ""

    try:
        pages: List[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                pages.append(text)

        pages = _deduplicate_headers_footers(pages)
        return "\n\n".join(p for p in pages if p.strip()).strip()

    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract plain text from a PDF resume.

    Tries PyMuPDF (block-sorted) first for layout-aware extraction, then
    pdfplumber as a fallback.  Applies ligature normalisation and delegates
    to preprocessing.clean_text() for further noise removal.

    Args:
        file_path (str): Absolute or relative path to a PDF file.

    Returns:
        str: Cleaned, whitespace-normalised text ready for downstream NLP.

    Raises:
        FileNotFoundError : if the file does not exist.
        ValueError        : if no extractor can produce non-empty text.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    # --- primary ---
    raw = _extract_with_pymupdf(file_path)

    # --- fallback ---
    if not raw.strip():
        raw = _extract_with_pdfplumber(file_path)

    if not raw.strip():
        raise ValueError(
            f"Could not extract text from '{path.name}'. "
            "The PDF may be image-only or encrypted."
        )

    # Normalise ligatures and unicode
    raw = _fix_ligatures(raw)

    # Delegate noise removal to the preprocessing pipeline
    try:
        from utils.preprocessing import clean_text  # noqa: PLC0415
        return clean_text(raw)
    except ImportError:
        # Graceful degradation if preprocessing module is not yet present
        return _basic_clean(raw)


def _basic_clean(text: str) -> str:
    """Minimal whitespace cleanup used when preprocessing module is unavailable."""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()
