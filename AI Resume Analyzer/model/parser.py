"""
parser.py
---------
Responsible for extracting raw text from a PDF resume using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> str:
    """
    Open a PDF file and extract all text content page by page.

    Args:
        file_path (str): Absolute or relative path to the PDF file.

    Returns:
        str: Concatenated text from all pages, stripped of leading/trailing whitespace.

    Raises:
        FileNotFoundError: If the PDF file does not exist at the given path.
        fitz.FileDataError: If the file is corrupted or not a valid PDF.
    """
    full_text = []

    with fitz.open(file_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_text = page.get_text("text")  # plain text extraction
            full_text.append(page_text)

    return "\n".join(full_text).strip()
