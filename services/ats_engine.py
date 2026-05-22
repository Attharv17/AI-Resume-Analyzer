"""
similarity_engine.py
--------------------
Semantic similarity computation using sentence embeddings.

Responsibilities
----------------
1.  Full-text semantic score  – cosine similarity between resume and JD
    embeddings, scaled to an integer percentage (0–100).
2.  Semantic skill matching   – for each job skill, finds the closest resume
    skill by embedding similarity; marks as matched if sim ≥ threshold.
3.  Confidence score          – mean of per-job-skill best-match similarities,
    giving an indication of how certain the matching is.

Public API
----------
    semantic_cosine_score(emb_a, emb_b)         -> float (0.0–1.0)
    semantic_skill_match(resume_skills,
                         job_skills,
                         threshold=0.75)         -> dict
    compute_semantic_match(resume_text, job_text,
                           resume_skills, job_skills) -> dict

Response shape from compute_semantic_match
------------------------------------------
    {
        "score":          int   (0-100),   # full-text semantic similarity
        "confidence":     float (0.0-1.0), # mean skill-level match confidence
        "matched_skills": List[str],       # semantically matched skills
        "missing_skills": List[str],       # job skills absent in resume
        "scoring_method": str,             # "semantic" | "tfidf_fallback"
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from services.model_service import encode, encode_single, expand_abbreviations

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

#: Cosine similarity threshold above which two skills are considered a match.
#: Range [0, 1]. Higher → stricter. 0.75 balances precision/recall well for
#: short skill strings encoded by MiniLM.
SKILL_MATCH_THRESHOLD: float = 0.75

#: If a job description or resume text is longer than this many characters,
#: truncate before embedding to keep latency acceptable.
MAX_TEXT_CHARS: int = 8_000


# ---------------------------------------------------------------------------
# Core cosine helpers
# ---------------------------------------------------------------------------

def semantic_cosine_score(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two L2-normalised embedding vectors.

    Because :func:`embedding_service.encode` returns L2-normalised vectors,
    cosine similarity reduces to a simple dot product.

    Args:
        emb_a: 1-D or 2-D numpy array (first embedding).
        emb_b: 1-D or 2-D numpy array (second embedding).

    Returns:
        float in [0, 1].
    """
    a = emb_a.flatten().astype(np.float32)
    b = emb_b.flatten().astype(np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Semantic skill matching
# ---------------------------------------------------------------------------

def semantic_skill_match(
    resume_skills: List[str],
    job_skills: List[str],
    threshold: float = SKILL_MATCH_THRESHOLD,
) -> Dict[str, Any]:
    """
    Match job skills against resume skills using semantic embeddings.

    For each job skill, find the highest-similarity resume skill. If that
    similarity exceeds *threshold* the job skill is considered MATCHED;
    otherwise it is MISSING.

    Handles abbreviation expansion internally (e.g. "js" → "javascript")
    so that lexically different but semantically equivalent terms are matched.

    Args:
        resume_skills : Skills extracted from the candidate's resume.
        job_skills    : Skills required by the job description.
        threshold     : Cosine similarity cutoff (default 0.75).

    Returns:
        {
            "matched_skills": List[str],   # job skills found in resume
            "missing_skills": List[str],   # job skills absent in resume
            "confidence":     float,       # mean best-match similarity (0–1)
            "skill_similarities": dict,    # job_skill → best cosine score
        }
    """
    # Edge cases
    if not job_skills:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "confidence": 1.0,
            "skill_similarities": {},
        }

    if not resume_skills:
        return {
            "matched_skills": [],
            "missing_skills": list(job_skills),
            "confidence": 0.0,
            "skill_similarities": {s: 0.0 for s in job_skills},
        }

    # Encode all resume skills in one batch
    resume_embeddings = encode(resume_skills)
    if resume_embeddings is None:
        # Fallback: exact string matching
        return _exact_skill_match(resume_skills, job_skills)

    matched: List[str] = []
    missing: List[str] = []
    similarities: Dict[str, float] = {}
    confidence_scores: List[float] = []

    for job_skill in job_skills:
        job_emb = encode_single(job_skill)
        if job_emb is None:
            # If a single skill encoding fails, do exact match for this skill
            if job_skill.lower() in {r.lower() for r in resume_skills}:
                matched.append(job_skill)
            else:
                missing.append(job_skill)
            similarities[job_skill] = 1.0 if job_skill in matched else 0.0
            confidence_scores.append(similarities[job_skill])
            continue

        # Compute similarity against every resume skill embedding
        # resume_embeddings is (N, 384); job_emb is (384,)
        sims = resume_embeddings @ job_emb  # dot product = cosine (L2-normed)
        best_sim = float(np.max(sims))
        best_idx = int(np.argmax(sims))

        similarities[job_skill] = round(best_sim, 4)
        confidence_scores.append(best_sim)

        if best_sim >= threshold:
            matched.append(job_skill)
            logger.debug(
                "MATCH  '%s' → '%s'  sim=%.3f",
                job_skill,
                resume_skills[best_idx],
                best_sim,
            )
        else:
            missing.append(job_skill)
            logger.debug(
                "MISS   '%s' best='%s'  sim=%.3f < threshold=%.2f",
                job_skill,
                resume_skills[best_idx],
                best_sim,
                threshold,
            )

    confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0

    return {
        "matched_skills":    sorted(matched),
        "missing_skills":    sorted(missing),
        "confidence":        round(confidence, 4),
        "skill_similarities": similarities,
    }


def _exact_skill_match(
    resume_skills: List[str],
    job_skills: List[str],
) -> Dict[str, Any]:
    """Exact (case-insensitive) string matching — used when embeddings fail."""
    resume_lower = {s.lower() for s in resume_skills}
    matched = [s for s in job_skills if s.lower() in resume_lower]
    missing = [s for s in job_skills if s.lower() not in resume_lower]
    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "confidence": len(matched) / max(len(job_skills), 1),
        "skill_similarities": {},
    }


# ---------------------------------------------------------------------------
# Main entry point — compute_semantic_match
# ---------------------------------------------------------------------------

def compute_semantic_match(
    resume_text: str,
    job_text: str,
    resume_skills: List[str],
    job_skills: List[str],
) -> Dict[str, Any]:
    """
    Compute the full semantic match between a resume and a job description.

    Steps
    -----
    1. Expand abbreviations in both texts.
    2. Encode full texts → compute full-text cosine similarity → *score*.
    3. Semantically match skill lists → *matched_skills*, *missing_skills*.
    4. Blend text score with skill-coverage score for a balanced *score*.
    5. Return structured result.

    Args:
        resume_text   : Cleaned full text extracted from the resume PDF.
        job_text      : Job description text (from the form or augmented).
        resume_skills : Skills extracted from the resume.
        job_skills    : Skills extracted from the job description.

    Returns:
        {
            "score":          int   (0-100),
            "confidence":     float (0.0-1.0),
            "matched_skills": List[str],
            "missing_skills": List[str],
            "scoring_method": str,   # "semantic" | "tfidf_fallback"
        }
    """
    # ------------------------------------------------------------------
    # 1. Attempt full-text semantic scoring
    # ------------------------------------------------------------------
    semantic_text_score: Optional[float] = None

    # Truncate long texts for embedding speed (model max is 256 tokens ≈ 1.5k chars)
    resume_snippet = resume_text[:MAX_TEXT_CHARS]
    job_snippet    = job_text[:MAX_TEXT_CHARS]

    resume_emb = encode_single(resume_snippet)
    job_emb    = encode_single(job_snippet)

    if resume_emb is not None and job_emb is not None:
        semantic_text_score = semantic_cosine_score(resume_emb, job_emb)
    else:
        # Embedding unavailable → fall back to TF-IDF
        logger.warning("Embedding unavailable; falling back to TF-IDF scoring.")
        return _tfidf_fallback(resume_text, job_text, resume_skills, job_skills)

    # ------------------------------------------------------------------
    # 2. Semantic skill matching
    # ------------------------------------------------------------------
    skill_result = semantic_skill_match(resume_skills, job_skills)

    matched_skills  = skill_result["matched_skills"]
    missing_skills  = skill_result["missing_skills"]
    confidence      = skill_result["confidence"]

    # ------------------------------------------------------------------
    # 3. Blended score
    #    70 % full-text semantic similarity
    #    30 % skill-coverage ratio
    # ------------------------------------------------------------------
    total_job_skills = len(job_skills)
    skill_coverage = (
        len(matched_skills) / total_job_skills if total_job_skills > 0 else 0.0
    )
    blended = 0.70 * semantic_text_score + 0.30 * skill_coverage
    final_score = round(min(max(blended * 100, 0), 100))

    return {
        "score":          final_score,
        "confidence":     round(confidence, 4),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "scoring_method": "semantic",
    }


# ---------------------------------------------------------------------------
# TF-IDF fallback (used if sentence-transformers is unavailable)
# ---------------------------------------------------------------------------

def _tfidf_fallback(
    resume_text: str,
    job_text: str,
    resume_skills: List[str],
    job_skills: List[str],
) -> Dict[str, Any]:
    """
    Fall back to the original TF-IDF scoring if embeddings are unavailable.

    This ensures the application is always functional even without the
    sentence-transformers package.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    matched = sorted(set(resume_skills) & set(job_skills))
    missing = sorted(set(job_skills) - set(resume_skills))

    score = 0
    if resume_text.strip() and job_text.strip():
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform([resume_text, job_text])
            sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
            score = round(min(max(float(sim) * 100, 0), 100))
        except Exception as exc:
            logger.error("TF-IDF fallback failed: %s", exc)

    confidence = (
        len(matched) / max(len(job_skills), 1) if job_skills else 0.0
    )

    return {
        "score":          score,
        "confidence":     round(confidence, 4),
        "matched_skills": matched,
        "missing_skills": missing,
        "scoring_method": "tfidf_fallback",
    }
