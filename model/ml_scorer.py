"""
ml_scorer.py (Phase 5)
----------------------
Scikit-learn model (RandomForestRegressor) mapping:
  [skill_match, experience_match, similarity_score] -> score in [0, 100]

Includes synthetic training data generation, train(), predict(), and save/load helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


FEATURE_NAMES = ["skill_match", "experience_match", "similarity_score"]


def _synthetic_dataset(n_samples: int = 600, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data with a known-ish rule + noise so the forest
    learns a smooth mapping to 0-100 targets.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, 3))
    sm, em, sim = X[:, 0], X[:, 1], X[:, 2]
    # Hidden-ish teacher signal (not identical to hybrid weights, on purpose)
    y = (
        38.0 * sm
        + 27.0 * em
        + 30.0 * sim
        + rng.normal(0, 4.0, size=n_samples)
    )
    y = np.clip(y, 0.0, 100.0)
    return X.astype(np.float32), y.astype(np.float32)


def train_model(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestRegressor:
    if X is None or y is None:
        X, y = _synthetic_dataset()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_scores(model: RandomForestRegressor, features: np.ndarray) -> np.ndarray:
    """
    features: shape (n, 3) in order FEATURE_NAMES
    Returns: (n,) floats in [0, 100] after clip
    """
    preds = model.predict(features.astype(np.float32))
    return np.clip(preds, 0.0, 100.0)


def features_dict_to_matrix(row: Dict[str, float]) -> np.ndarray:
    return np.array(
        [[row["skill_match"], row["experience_match"], row["similarity_score"]]],
        dtype=np.float32,
    )


def save_model(model: RandomForestRegressor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> RandomForestRegressor:
    return joblib.load(path)


def ensure_default_model(models_dir: str | Path | None = None) -> RandomForestRegressor:
    """
    Load models/score_model.joblib if present; otherwise train, save, and return.
    """
    base = Path(models_dir) if models_dir else Path(__file__).resolve().parent.parent / "model_store"
    mpath = base / "score_model.joblib"
    if mpath.is_file():
        return load_model(mpath)
    model = train_model()
    save_model(model, mpath)
    return model
