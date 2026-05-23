"""
ats_engine.py  (formerly similarity_engine.py)
-----------------------------------------------
Semantic similarity computation using sentence embeddings.

Responsibilities
----------------
1.  Full-text semantic score  – cosine similarity between resume and JD
    embeddings, stretched via sigmoid to use the full [0, 1] range.
2.  Exact skill matching      – fast set-intersection for crisp overlap signal.
3.  Semantic skill matching   – for each job skill, finds the closest resume
    skill by embedding similarity; marks as matched if sim ≥ threshold.
4.  Penalty & boost engine    – penalises missing critical skills, rewards
    strong alignment, ensures realistic ATS score distributions.
5.  Debug logging             – every component is logged so you can see
    exactly why a score landed where it did.

Score Distribution Targets
---------------------------
  Excellent resume  →  85–95 %
  Good resume       →  70–85 %
  Average resume    →  45–70 %
  Weak resume       →  < 45 %

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
        "score":          int   (0-100),
        "confidence":     float (0.0-1.0),
        "matched_skills": List[str],
        "missing_skills": List[str],
        "scoring_method": str,          # "semantic" | "tfidf_fallback"
        "score_breakdown": dict,        # debug: each component value
    }
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from services.model_service import encode, encode_single, expand_abbreviations

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

#: Cosine similarity threshold above which two skills are considered a match.
SKILL_MATCH_THRESHOLD: float = 0.75

#: If a job description or resume text is longer than this many characters,
#: truncate before embedding to keep latency acceptable.
MAX_TEXT_CHARS: int = 8_000

# Sigmoid stretch parameters — pull the compressed 0.55–0.80 cosine band
# out into a full 0.0–1.0 range.
#   center    : the raw cosine value that maps to 0.50 stretched
_SIG_CENTER: float = 0.45

#: How aggressively to separate weak from strong cosine values.
#: Lower values give a softer S-curve; higher values are more binary.
_SIG_STEEPNESS: float = 5.0

# Component weights (must sum to 1.0)
_W_SEMANTIC: float = 0.35   # stretched full-text cosine similarity
_W_EXACT:    float = 0.40   # exact (set-intersection) skill overlap
_W_SEMSKILL: float = 0.25   # semantic skill coverage ratio

# Penalty caps
_MAX_PENALTY: float = 0.25

# Boost caps
_MAX_BOOST: float = 0.12

# Minimum final score (prevents 0% for any resume that has skills)
_MIN_SCORE_WITH_SKILLS: int = 3


# ---------------------------------------------------------------------------
# Sigmoid stretch — breaks cosine compression
# ---------------------------------------------------------------------------

def _sigmoid_stretch(x: float, center: float = _SIG_CENTER,
                     steepness: float = _SIG_STEEPNESS) -> float:
    """
    Map raw cosine similarity to a stretched [0, 1] value.

    The all-MiniLM model clusters any two domain-related texts between
    0.55 and 0.80 — this function spreads that band across the full range so
    weak and strong resumes receive meaningfully different scores.

    Examples (center=0.62, steepness=10):
        raw 0.40 → ~0.10  (very weak)
        raw 0.55 → ~0.27  (weak)
        raw 0.62 → ~0.50  (average baseline)
        raw 0.70 → ~0.69  (good)
        raw 0.78 → ~0.85  (strong)
        raw 0.85 → ~0.94  (excellent)
    """
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


# ---------------------------------------------------------------------------
# Core cosine helpers
# ---------------------------------------------------------------------------

def semantic_cosine_score(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two L2-normalised embedding vectors.

    Because encode() returns L2-normalised vectors, cosine similarity
    reduces to a simple dot product.

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
# Exact skill matching (set intersection — high variance signal)
# ---------------------------------------------------------------------------

def _exact_overlap_ratio(
    resume_skills: List[str],
    job_skills: List[str],
) -> float:
    """
    Compute the fraction of job skills that appear EXACTLY in the resume
    (case-insensitive set intersection).

    This is the highest-variance signal because it directly measures whether
    the candidate has the required technologies without any cosine smoothing.

    Returns:
        float in [0.0, 1.0]
    """
    if not job_skills:
        return 1.0
    if not resume_skills:
        return 0.0
    resume_lower = {s.strip().lower() for s in resume_skills}
    matched = sum(1 for s in job_skills if s.strip().lower() in resume_lower)
    return matched / len(job_skills)


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

    Args:
        resume_skills : Skills extracted from the candidate's resume.
        job_skills    : Skills required by the job description.
        threshold     : Cosine similarity cutoff (default 0.75).

    Returns:
        {
            "matched_skills": List[str],
            "missing_skills": List[str],
            "confidence":     float,
            "skill_similarities": dict,
        }
    """
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

    resume_embeddings = encode(resume_skills)
    if resume_embeddings is None:
        return _exact_skill_match_fallback(resume_skills, job_skills)

    matched: List[str] = []
    missing: List[str] = []
    similarities: Dict[str, float] = {}
    confidence_scores: List[float] = []

    for job_skill in job_skills:
        job_emb = encode_single(job_skill)
        if job_emb is None:
            if job_skill.lower() in {r.lower() for r in resume_skills}:
                matched.append(job_skill)
            else:
                missing.append(job_skill)
            similarities[job_skill] = 1.0 if job_skill in matched else 0.0
            confidence_scores.append(similarities[job_skill])
            continue

        sims = resume_embeddings @ job_emb
        best_sim = float(np.max(sims))
        best_idx = int(np.argmax(sims))

        similarities[job_skill] = round(best_sim, 4)
        confidence_scores.append(best_sim)

        if best_sim >= threshold:
            matched.append(job_skill)
            logger.debug(
                "MATCH  '%s' → '%s'  sim=%.3f",
                job_skill, resume_skills[best_idx], best_sim,
            )
        else:
            missing.append(job_skill)
            logger.debug(
                "MISS   '%s' best='%s'  sim=%.3f < threshold=%.2f",
                job_skill, resume_skills[best_idx], best_sim, threshold,
            )

    confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0

    return {
        "matched_skills":     sorted(matched),
        "missing_skills":     sorted(missing),
        "confidence":         round(confidence, 4),
        "skill_similarities": similarities,
    }


def _exact_skill_match_fallback(
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
# Penalty & boost engine
# ---------------------------------------------------------------------------

def _compute_penalties(
    exact_overlap: float,
    semantic_skill_coverage: float,
    missing_skills: List[str],
    total_job_skills: int,
    resume_skills: List[str],
) -> tuple[float, list[str]]:
    """
    Compute penalty deductions based on resume weaknesses.

    Returns:
        (total_penalty: float, reasons: list[str])
    """
    penalty = 0.0
    reasons: list[str] = []

    # --- Penalty 1: Large fraction of job skills semantically missing ---
    missing_ratio = len(missing_skills) / max(total_job_skills, 1)
    if missing_ratio >= 0.75:
        p = 0.08
        penalty += p
        reasons.append(f"critical_skill_gap({p:.2f}): {missing_ratio:.0%} of required skills missing")
    elif missing_ratio >= 0.55:
        p = 0.06
        penalty += p
        reasons.append(f"skill_gap({p:.2f}): {missing_ratio:.0%} of required skills missing")
    elif missing_ratio >= 0.40:
        p = 0.02
        penalty += p
        reasons.append(f"partial_skill_gap({p:.2f}): {missing_ratio:.0%} of required skills missing")

    # --- Penalty 2: Near-zero exact overlap (completely wrong domain) ---
    if exact_overlap < 0.10 and total_job_skills >= 5:
        p = 0.05
        penalty += p
        reasons.append(f"wrong_domain({p:.2f}): exact overlap only {exact_overlap:.0%}")
    elif exact_overlap < 0.20 and total_job_skills >= 5:
        p = 0.03
        penalty += p
        reasons.append(f"low_domain_fit({p:.2f}): exact overlap only {exact_overlap:.0%}")

    # --- Penalty 3: Very sparse resume skills (< 5 skills is genuinely thin) ---
    if len(resume_skills) < 3:
        p = 0.06
        penalty += p
        reasons.append(f"sparse_resume({p:.2f}): only {len(resume_skills)} skills extracted")
    elif len(resume_skills) < 5:
        p = 0.03
        penalty += p
        reasons.append(f"thin_resume({p:.2f}): only {len(resume_skills)} skills extracted")

    return min(penalty, _MAX_PENALTY), reasons


def _compute_boosts(
    exact_overlap: float,
    semantic_skill_coverage: float,
    stretched_sim: float,
) -> tuple[float, list[str]]:
    """
    Compute score boosts for strong alignment signals.

    Returns:
        (total_boost: float, reasons: list[str])
    """
    boost = 0.0
    reasons: list[str] = []

    # --- Boost 1: Near-perfect exact skill overlap ---
    if exact_overlap >= 0.85:
        b = 0.10
        boost += b
        reasons.append(f"perfect_skill_match({b:.2f}): {exact_overlap:.0%} exact overlap")
    elif exact_overlap >= 0.65:
        b = 0.05
        boost += b
        reasons.append(f"strong_skill_match({b:.2f}): {exact_overlap:.0%} exact overlap")

    # --- Boost 2: Very strong semantic alignment ---
    if stretched_sim >= 0.80:
        b = 0.05
        boost += b
        reasons.append(f"strong_semantic({b:.2f}): stretched_sim={stretched_sim:.3f}")

    # --- Boost 3: High semantic skill coverage ---
    if semantic_skill_coverage >= 0.80:
        b = 0.04
        boost += b
        reasons.append(f"high_semantic_coverage({b:.2f}): {semantic_skill_coverage:.0%}")

    return min(boost, _MAX_BOOST), reasons


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

    Scoring Formula
    ---------------
    1. Stretch raw cosine similarity through a sigmoid to break compression.
    2. Compute exact skill overlap via set intersection (highest variance).
    3. Compute semantic skill coverage (embedding-based matching).
    4. Blend three components with tuned weights.
    5. Apply penalties for missing skills / wrong domain / sparse resume.
    6. Apply boosts for strong alignment.
    7. Scale to 0–100 and log every component for debugging.

    Args:
        resume_text   : Cleaned full text extracted from the resume PDF.
        job_text      : Job description text.
        resume_skills : Skills extracted from the resume.
        job_skills    : Skills extracted from the job description.

    Returns:
        {
            "score":          int   (0-100),
            "confidence":     float (0.0-1.0),
            "matched_skills": List[str],
            "missing_skills": List[str],
            "scoring_method": str,
            "score_breakdown": dict,
        }
    """
    # ------------------------------------------------------------------
    # 1. Attempt full-text semantic scoring
    # ------------------------------------------------------------------
    resume_snippet = resume_text[:MAX_TEXT_CHARS]
    job_snippet    = job_text[:MAX_TEXT_CHARS]

    resume_emb = encode_single(resume_snippet)
    job_emb    = encode_single(job_snippet)

    if resume_emb is None or job_emb is None:
        logger.warning("Embedding unavailable; falling back to TF-IDF scoring.")
        return _tfidf_fallback(resume_text, job_text, resume_skills, job_skills)

    raw_cosine    = semantic_cosine_score(resume_emb, job_emb)
    stretched_sim = _sigmoid_stretch(raw_cosine)

    # ------------------------------------------------------------------
    # 2. Exact skill overlap (set intersection — high variance)
    # ------------------------------------------------------------------
    exact_overlap = _exact_overlap_ratio(resume_skills, job_skills)

    # ------------------------------------------------------------------
    # 3. Semantic skill matching
    # ------------------------------------------------------------------
    skill_result = semantic_skill_match(resume_skills, job_skills)

    matched_skills  = skill_result["matched_skills"]
    missing_skills  = skill_result["missing_skills"]
    confidence      = skill_result["confidence"]

    total_job_skills = len(job_skills)
    semantic_skill_coverage = (
        len(matched_skills) / total_job_skills if total_job_skills > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # 4. Weighted base score
    # ------------------------------------------------------------------
    base_score = (
        _W_SEMANTIC * stretched_sim
        + _W_EXACT   * exact_overlap
        + _W_SEMSKILL * semantic_skill_coverage
    )

    # ------------------------------------------------------------------
    # 5. Penalties
    # ------------------------------------------------------------------
    penalty, penalty_reasons = _compute_penalties(
        exact_overlap=exact_overlap,
        semantic_skill_coverage=semantic_skill_coverage,
        missing_skills=missing_skills,
        total_job_skills=total_job_skills,
        resume_skills=resume_skills,
    )

    # ------------------------------------------------------------------
    # 6. Boosts
    # ------------------------------------------------------------------
    boost, boost_reasons = _compute_boosts(
        exact_overlap=exact_overlap,
        semantic_skill_coverage=semantic_skill_coverage,
        stretched_sim=stretched_sim,
    )

    # ------------------------------------------------------------------
    # 7. Final score — apply floor so no resume with skills gets 0%
    # ------------------------------------------------------------------
    adjusted = base_score - penalty + boost
    raw_final = adjusted * 100

    # Apply minimum floor for resumes that have *some* skills
    if resume_skills:
        raw_final = max(raw_final, _MIN_SCORE_WITH_SKILLS)

    final_score = int(round(min(max(raw_final, 0), 100)))

    # ------------------------------------------------------------------
    # 8. Debug logging
    # ------------------------------------------------------------------
    logger.info(
        "[ATS DEBUG] raw_cosine=%.3f  stretched_sim=%.3f",
        raw_cosine, stretched_sim,
    )
    logger.info(
        "[ATS DEBUG] exact_overlap=%.3f  semantic_coverage=%.3f",
        exact_overlap, semantic_skill_coverage,
    )
    logger.info(
        "[ATS DEBUG] base_score=%.3f  (w_sem=%.2f w_exact=%.2f w_semskill=%.2f)",
        base_score, _W_SEMANTIC * stretched_sim,
        _W_EXACT * exact_overlap, _W_SEMSKILL * semantic_skill_coverage,
    )
    if penalty_reasons:
        logger.info("[ATS DEBUG] penalties=-%.3f  %s", penalty, " | ".join(penalty_reasons))
    else:
        logger.info("[ATS DEBUG] penalties=0.000  (none applied)")
    if boost_reasons:
        logger.info("[ATS DEBUG] boosts=+%.3f  %s", boost, " | ".join(boost_reasons))
    else:
        logger.info("[ATS DEBUG] boosts=0.000  (none applied)")
    logger.info(
        "[ATS DEBUG] final_score=%d  matched=%d/%d  missing=%d",
        final_score, len(matched_skills), total_job_skills, len(missing_skills),
    )

    breakdown = {
        "raw_cosine":            round(raw_cosine, 4),
        "stretched_sim":         round(stretched_sim, 4),
        "exact_overlap":         round(exact_overlap, 4),
        "semantic_coverage":     round(semantic_skill_coverage, 4),
        "base_score":            round(base_score, 4),
        "penalty":               round(penalty, 4),
        "boost":                 round(boost, 4),
        "penalty_reasons":       penalty_reasons,
        "boost_reasons":         boost_reasons,
        "weights": {
            "semantic":  _W_SEMANTIC,
            "exact":     _W_EXACT,
            "sem_skill": _W_SEMSKILL,
        },
    }

    return {
        "score":          final_score,
        "confidence":     round(confidence, 4),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "scoring_method": "semantic",
        "score_breakdown": breakdown,
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
    Fall back to TF-IDF scoring if embeddings are unavailable.
    Also applies the sigmoid stretch so the score range is calibrated
    consistently with the embedding path.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    matched = sorted(set(resume_skills) & set(job_skills))
    missing = sorted(set(job_skills) - set(resume_skills))

    raw_cosine = 0.0
    if resume_text.strip() and job_text.strip():
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform([resume_text, job_text])
            raw_cosine = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        except Exception as exc:
            logger.error("TF-IDF fallback failed: %s", exc)

    # Apply the same sigmoid stretch for calibration consistency
    stretched_sim = _sigmoid_stretch(raw_cosine)
    exact_overlap = _exact_overlap_ratio(resume_skills, job_skills)
    total_job_skills = len(job_skills)
    semantic_skill_coverage = len(matched) / max(total_job_skills, 1)

    base_score = (
        _W_SEMANTIC  * stretched_sim
        + _W_EXACT   * exact_overlap
        + _W_SEMSKILL * semantic_skill_coverage
    )

    penalty, penalty_reasons = _compute_penalties(
        exact_overlap=exact_overlap,
        semantic_skill_coverage=semantic_skill_coverage,
        missing_skills=missing,
        total_job_skills=total_job_skills,
        resume_skills=resume_skills,
    )
    boost, boost_reasons = _compute_boosts(
        exact_overlap=exact_overlap,
        semantic_skill_coverage=semantic_skill_coverage,
        stretched_sim=stretched_sim,
    )

    final_score = int(round(min(max((base_score - penalty + boost) * 100, 0), 100)))

    logger.info(
        "[ATS DEBUG tfidf] raw_cosine=%.3f stretched=%.3f exact=%.3f final=%d",
        raw_cosine, stretched_sim, exact_overlap, final_score,
    )

    confidence = len(matched) / max(total_job_skills, 1)
    breakdown = {
        "raw_cosine": round(raw_cosine, 4),
        "stretched_sim": round(stretched_sim, 4),
        "exact_overlap": round(exact_overlap, 4),
        "semantic_coverage": round(semantic_skill_coverage, 4),
        "base_score": round(base_score, 4),
        "penalty": round(penalty, 4),
        "boost": round(boost, 4),
        "penalty_reasons": penalty_reasons,
        "boost_reasons": boost_reasons,
    }

    return {
        "score":          final_score,
        "confidence":     round(confidence, 4),
        "matched_skills": matched,
        "missing_skills": missing,
        "scoring_method": "tfidf_fallback",
        "score_breakdown": breakdown,
    }
