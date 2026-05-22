"""
hybrid_scorer.py (Phase 6)
--------------------------
Combine rule-based features with ML model prediction:

Final =
  0.4 * skill_match +
  0.3 * experience_match +
  0.2 * similarity +
  0.1 * (model_prediction / 100)

Returns final score in [0, 1] (or scale to 0-100 via parameter).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from sklearn.ensemble import RandomForestRegressor

from model.ml_scorer import predict_scores


def hybrid_final_score(
    skill_match: float,
    experience_match: float,
    similarity: float,
    model: RandomForestRegressor,
    scale_0_100: bool = False,
) -> float:
    """
    All inputs in [0,1] except model which predicts [0,100].
    """
    x = np_array_row(skill_match, experience_match, similarity)
    ml = float(predict_scores(model, x)[0])
    ml_norm = ml / 100.0
    base = (
        0.4 * float(skill_match)
        + 0.3 * float(experience_match)
        + 0.2 * float(similarity)
        + 0.1 * ml_norm
    )
    out = max(0.0, min(1.0, base))
    return out * 100.0 if scale_0_100 else out


def np_array_row(a: float, b: float, c: float):
    import numpy as np

    return np.array([[a, b, c]], dtype=np.float32)


def hybrid_from_feature_dict(
    fv: Dict[str, float],
    model: RandomForestRegressor,
    scale_0_100: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (final_score, debug_dict with ml raw prediction)
    """
    sm = float(fv["skill_match"])
    em = float(fv["experience_match"])
    sim = float(fv["similarity_score"])
    x = np_array_row(sm, em, sim)
    ml_raw = float(predict_scores(model, x)[0])
    ml_norm = ml_raw / 100.0
    combined = (
        0.4 * sm
        + 0.3 * em
        + 0.2 * sim
        + 0.1 * ml_norm
    )
    combined = max(0.0, min(1.0, combined))
    final = combined * 100.0 if scale_0_100 else combined
    debug = {
        "skill_match": sm,
        "experience_match": em,
        "similarity_score": sim,
        "ml_prediction_0_100": ml_raw,
        "final_0_100": final if scale_0_100 else combined * 100.0,
    }
    return final, debug
