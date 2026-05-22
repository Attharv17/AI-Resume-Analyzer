"""
matcher.py
----------
Computes the match between a resume and a job description using two approaches:

1. **Semantic score** (TF-IDF + cosine similarity)
   - Transforms the full resume text and full job-description text into
     TF-IDF vectors, then computes the cosine similarity.
   - This captures natural-language overlap beyond exact keyword hits.

2. **Skill gap analysis** (set intersection / difference)
   - Uses the skill lists already extracted by extractor.py.
   - Identifies which job skills are present / absent in the resume.

Final response shape:
    {
        "score":          int (0-100),
        "matched_skills": List[str],
        "missing_skills": List[str],
    }
"""

from typing import Dict, Any, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_match(
    resume_text: str,
    job_text: str,
    resume_skills: List[str],
    job_skills: List[str],
) -> Dict[str, Any]:
    """
    Compute the overall match between a resume and a job description.

    Args:
        resume_text   (str)       : Raw extracted text from the resume PDF.
        job_text      (str)       : Raw job description text (from the form).
        resume_skills (List[str]) : Skills extracted from the resume.
        job_skills    (List[str]) : Skills extracted from the job description.

    Returns:
        Dict with keys:
            "score"          – TF-IDF cosine similarity scaled to 0-100 (int).
            "matched_skills" – Skills found in both resume and job description.
            "missing_skills" – Job skills absent from the resume.
    """
    # ------------------------------------------------------------------
    # 1. Semantic score via TF-IDF + cosine similarity
    # ------------------------------------------------------------------
    semantic_score = _tfidf_cosine_score(resume_text, job_text)

    # ------------------------------------------------------------------
    # 2. Skill gap analysis (keyword-level)
    # ------------------------------------------------------------------
    resume_set = set(resume_skills)
    job_set    = set(job_skills)

    matched: List[str] = sorted(resume_set & job_set)
    missing: List[str] = sorted(job_set - resume_set)

    return {
        "score":          semantic_score,
        "matched_skills": matched,
        "missing_skills": missing,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tfidf_cosine_score(text_a: str, text_b: str) -> int:
    """
    Compute cosine similarity between two documents using TF-IDF vectors.

    Steps:
      1. Fit a TfidfVectorizer on both documents simultaneously so that the
         vocabulary (IDF weights) reflects the union of both texts.
      2. Transform each document into its TF-IDF vector.
      3. Compute cosine similarity (a value in [0, 1]).
      4. Scale to an integer percentage (0-100).

    Args:
        text_a (str): First document (resume text).
        text_b (str): Second document (job description text).

    Returns:
        int: Similarity percentage between 0 and 100.
    """
    if not text_a.strip() or not text_b.strip():
        return 0

    vectorizer = TfidfVectorizer(
        stop_words="english",      # remove common English stop words
        ngram_range=(1, 2),        # unigrams + bigrams for phrase-level matching
        min_df=1,
        sublinear_tf=True,         # apply log(1+tf) to dampen high-frequency terms
    )

    # Fit on both documents so IDF is computed over the combined vocabulary
    tfidf_matrix = vectorizer.fit_transform([text_a, text_b])

    # cosine_similarity returns a (1×1 or 2×2) matrix; we want [0,1] entry
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Clamp to [0, 1] (float precision can produce values like 1.0000000002)
    similarity = max(0.0, min(1.0, float(similarity)))

    return round(similarity * 100)
