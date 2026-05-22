"""
embedding_service.py
--------------------
Singleton wrapper around sentence-transformers/all-MiniLM-L6-v2.

Responsibilities
----------------
1. Lazy model loading — model is downloaded and cached on first call, then
   reused for every subsequent request (no per-request overhead).
2. Batch encoding — encodes a list of strings into L2-normalised embeddings.
3. Abbreviation expansion — expands short tokens BEFORE encoding so that
   embeddings for "js", "ml", "ai", "node" are semantically accurate.

Public API
----------
    get_model()  -> SentenceTransformer | None
    encode(texts, batch_size=32)  -> np.ndarray   shape (N, 384)
    encode_single(text)           -> np.ndarray   shape (384,)
    expand_abbreviations(text)    -> str
    is_available()                -> bool
"""

from __future__ import annotations

import logging
import re
import threading
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abbreviation expansion map
# Tokens are matched as whole words (word-boundary aware).
# Applied BEFORE encoding to improve embedding quality of short strings.
# ---------------------------------------------------------------------------

_ABBR_MAP: dict[str, str] = {
    # Language / framework abbreviations
    "js":          "javascript",
    "ts":          "typescript",
    "py":          "python",
    # ML / AI abbreviations
    "ml":          "machine learning",
    "dl":          "deep learning",
    "ai":          "artificial intelligence",
    "nlp":         "natural language processing",
    "cv":          "computer vision",
    "rl":          "reinforcement learning",
    "llm":         "large language model",
    "llms":        "large language models",
    "gen ai":      "generative artificial intelligence",
    "genai":       "generative artificial intelligence",
    "rag":         "retrieval augmented generation",
    # Node / runtime aliases
    "node":        "node.js",
    # Cloud / infra abbreviations
    "k8s":         "kubernetes",
    "kube":        "kubernetes",
    "tf":          "tensorflow",
    "gcp":         "google cloud platform",
    # Data abbreviations
    "bi":          "business intelligence",
    "etl":         "extract transform load",
    # Misc
    "oop":         "object oriented programming",
    "fp":          "functional programming",
    "ci cd":       "continuous integration continuous deployment",
    "cicd":        "continuous integration continuous deployment",
    "rest":        "representational state transfer rest api",
    "sql":         "structured query language sql",
}

# Pre-compile each pattern once at import time
_ABBR_PATTERNS: list[tuple[re.Pattern, str]] = []
for _alias, _expansion in _ABBR_MAP.items():
    _escaped = re.escape(_alias)
    # Use word-boundary for purely alphabetic aliases, lookaround for others
    if re.search(r"[^a-z]", _alias):
        _pat = r"(?<![a-z0-9])" + _escaped + r"(?![a-z0-9])"
    else:
        _pat = r"\b" + _escaped + r"\b"
    _ABBR_PATTERNS.append((re.compile(_pat, re.IGNORECASE), _expansion))


def expand_abbreviations(text: str) -> str:
    """
    Expand known abbreviations in *text* to their full canonical forms.

    Examples
    --------
    >>> expand_abbreviations("js and ml developer")
    'javascript and machine learning developer'
    >>> expand_abbreviations("Node developer with AI skills")
    'node.js developer with artificial intelligence skills'
    """
    for pattern, expansion in _ABBR_PATTERNS:
        text = pattern.sub(expansion, text)
    return text


# ---------------------------------------------------------------------------
# Singleton model holder
# ---------------------------------------------------------------------------

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None          # SentenceTransformer instance or False (unavailable)
_model_lock = threading.Lock()


def is_available() -> bool:
    """Return True if sentence-transformers can be imported."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def get_model():
    """
    Return the loaded SentenceTransformer singleton, loading it on first call.

    Returns None if sentence-transformers is not installed.
    Thread-safe via a module-level lock.
    """
    global _model

    # Fast path — already decided
    if _model is not None:
        return _model if _model is not False else None

    with _model_lock:
        # Double-checked locking
        if _model is not None:
            return _model if _model is not False else None

        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            logger.info("Loading embedding model: %s", _MODEL_NAME)
            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("Embedding model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "sentence-transformers not available (%s). "
                "Falling back to TF-IDF scoring.",
                exc,
            )
            _model = False

    return _model if _model is not False else None


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode(texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
    """
    Encode a list of strings into L2-normalised sentence embeddings.

    Each string is first processed through :func:`expand_abbreviations` so
    that abbreviations like "js" or "ml" produce accurate dense vectors.

    Args:
        texts      : List of strings to encode.
        batch_size : Sentences per encoding batch (default 32).

    Returns:
        np.ndarray of shape (len(texts), 384), or None if model unavailable.
    """
    model = get_model()
    if model is None or not texts:
        return None

    # Expand abbreviations before encoding
    expanded = [expand_abbreviations(t) for t in texts]

    try:
        embeddings = model.encode(
            expanded,
            batch_size=batch_size,
            normalize_embeddings=True,   # L2-normalise → cosine = dot product
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)
    except Exception as exc:
        logger.error("Encoding failed: %s", exc)
        return None


def encode_single(text: str) -> Optional[np.ndarray]:
    """
    Encode a single string into a 1-D embedding vector.

    Args:
        text: The string to encode.

    Returns:
        np.ndarray of shape (384,), or None if model unavailable.
    """
    result = encode([text], batch_size=1)
    if result is None:
        return None
    return result[0]
