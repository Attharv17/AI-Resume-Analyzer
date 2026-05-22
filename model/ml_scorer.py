"""
ml_scorer.py (Phase 5 — optimised)
------------------------------------
Scikit-learn / XGBoost model management with one-time startup loading.

Key improvements over the previous version
-------------------------------------------
1. **Unified model registry** — a single ``ModelRegistry`` object owns all
   artefact handles (RandomForestRegressor score model, legacy XGBRegressor
   resume model, legacy TfidfVectorizer) so nothing is loaded more than once
   per process lifetime.

2. **Thread-safe initialisation** — a ``threading.Lock`` guards first-load;
   subsequent calls take the fast path without acquiring the lock (double-
   checked locking pattern).

3. **Startup warm-up helper** — ``warm_up()`` is called once at app boot
   (Flask ``before_first_request`` / FastAPI ``lifespan``) so the first real
   request is never penalised by 3–4 second loads.

4. **Error isolation** — each artefact is loaded in its own try/except block;
   a missing or corrupt file leaves other models intact.  ``get_registry()``
   always returns a valid (possibly partially loaded) registry.

5. **Fixed missing import** — ``Dict`` is now imported from ``typing``
   (previous version raised ``NameError`` on ``features_dict_to_matrix``).

Public API (backward-compatible)
---------------------------------
    warm_up(models_dir)                    → None  (call once at startup)
    get_registry()                         → ModelRegistry
    predict_scores(model, features)        → np.ndarray
    train_model(X, y, ...)                 → RandomForestRegressor
    save_model(model, path)                → None
    load_model(path)                       → RandomForestRegressor
    ensure_default_model(models_dir)       → RandomForestRegressor  (kept for compat)
    features_dict_to_matrix(row)           → np.ndarray
"""

from __future__ import annotations

import logging
import threading
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["skill_match", "experience_match", "similarity_score"]

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_BASE_DIR   = Path(__file__).resolve().parent.parent
_MODEL_DIR  = _BASE_DIR / "model_store"
_LEGACY_DIR = _BASE_DIR / "model"

_DEFAULT_SCORE_MODEL_PATH    = _MODEL_DIR  / "score_model.joblib"
_LEGACY_RESUME_MODEL_PATH    = _LEGACY_DIR / "resume_model.pkl"
_LEGACY_VECTORIZER_PATH      = _LEGACY_DIR / "vectorizer.pkl"


# ---------------------------------------------------------------------------
# ModelRegistry — holds all loaded artefacts
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Centralised store for all ML model artefacts.

    All attributes default to ``None``; ``warm_up()`` fills them in.
    Consumers should check ``registry.score_model is not None`` before use.
    """

    def __init__(self) -> None:
        # Primary scoring model (RandomForestRegressor — score_model.joblib)
        self.score_model: Optional[RandomForestRegressor] = None

        # Legacy artefacts (resume_model.pkl / vectorizer.pkl)
        self.resume_model: Optional[object] = None   # XGBRegressor
        self.vectorizer:   Optional[object] = None   # TfidfVectorizer

        # Metadata
        self.score_model_path: Optional[Path] = None
        self._loaded: bool = False

    def is_ready(self) -> bool:
        """Return True if the primary scoring model is available."""
        return self.score_model is not None

    def status(self) -> Dict[str, bool]:
        return {
            "score_model":   self.score_model  is not None,
            "resume_model":  self.resume_model is not None,
            "vectorizer":    self.vectorizer   is not None,
        }


# ---------------------------------------------------------------------------
# Singleton registry + lock
# ---------------------------------------------------------------------------

_registry: Optional[ModelRegistry] = None
_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    """
    Return the process-level ModelRegistry, creating it if needed.

    This does NOT trigger loading — call :func:`warm_up` to load artefacts.
    """
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = ModelRegistry()
    return _registry


# ---------------------------------------------------------------------------
# Artefact loading helpers (each isolated in its own try/except)
# ---------------------------------------------------------------------------

def _load_score_model(path: Path) -> Optional[RandomForestRegressor]:
    """Load the primary RandomForest scoring model from *path*."""
    if not path.is_file():
        logger.warning("score_model not found at %s — will train a new one.", path)
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(path)
        logger.info("score_model loaded from %s", path)
        return model
    except Exception as exc:
        logger.error("Failed to load score_model from %s: %s", path, exc)
        return None


def _load_legacy_resume_model(path: Path) -> Optional[object]:
    """Load the legacy XGBRegressor resume_model.pkl if present."""
    if not path.is_file():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(path)
        logger.info("legacy resume_model loaded from %s", path)
        return model
    except Exception as exc:
        logger.warning("Could not load legacy resume_model (%s): %s", path, exc)
        return None


def _load_legacy_vectorizer(path: Path) -> Optional[object]:
    """Load the legacy TfidfVectorizer vectorizer.pkl if present."""
    if not path.is_file():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vec = joblib.load(path)
        logger.info("legacy vectorizer loaded from %s", path)
        return vec
    except Exception as exc:
        logger.warning("Could not load legacy vectorizer (%s): %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Synthetic training data + model training (unchanged from original)
# ---------------------------------------------------------------------------

def _synthetic_dataset(
    n_samples: int = 600, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data so the forest learns a smooth mapping
    to 0-100 targets.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, 3))
    sm, em, sim = X[:, 0], X[:, 1], X[:, 2]
    y = (
        38.0 * sm
        + 27.0 * em
        + 30.0 * sim
        + rng.normal(0, 4.0, size=n_samples)
    )
    y = np.clip(y, 0.0, 100.0)
    return X.astype(np.float32), y.astype(np.float32)


def train_model(
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a RandomForestRegressor, using synthetic data if none provided."""
    if X is None or y is None:
        X, y = _synthetic_dataset()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Public warm-up — call ONCE at application startup
# ---------------------------------------------------------------------------

def warm_up(models_dir: Optional[Path] = None) -> ModelRegistry:
    """
    Load all ML artefacts into the singleton registry.

    Designed to be called **once** at application startup (Flask/FastAPI boot).
    Subsequent calls are no-ops — the registry is only built once per process.

    Args:
        models_dir: Directory containing ``score_model.joblib``.
                    Defaults to ``<project_root>/model_store``.

    Returns:
        The populated :class:`ModelRegistry`.
    """
    global _registry
    reg = get_registry()

    # Fast path — already loaded, skip entirely
    if reg._loaded:
        return reg

    with _lock:
        # Double-checked inside the lock
        if reg._loaded:
            return reg

        logger.info("=== ML model warm-up starting ===")

        # ── 1. Primary score model ────────────────────────────────────────
        score_path = (
            Path(models_dir) / "score_model.joblib"
            if models_dir
            else _DEFAULT_SCORE_MODEL_PATH
        )
        model = _load_score_model(score_path)

        if model is None:
            # Train + save a fresh model so next startup is instant
            logger.info("Training new score_model (first-time setup)…")
            try:
                model = train_model()
                save_model(model, score_path)
                logger.info("score_model saved to %s", score_path)
            except Exception as exc:
                logger.error("Training failed: %s", exc)

        reg.score_model = model
        reg.score_model_path = score_path

        # ── 2. Legacy artefacts (non-blocking — failures are warnings) ────
        reg.resume_model = _load_legacy_resume_model(_LEGACY_RESUME_MODEL_PATH)
        reg.vectorizer   = _load_legacy_vectorizer(_LEGACY_VECTORIZER_PATH)

        reg._loaded = True
        logger.info("=== ML model warm-up complete. Status: %s ===", reg.status())

    return reg


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_scores(
    model: RandomForestRegressor,
    features: np.ndarray,
) -> np.ndarray:
    """
    Run inference through *model*.

    Args:
        model    : A fitted RandomForestRegressor (or compatible estimator).
        features : Shape (n, 3) in order FEATURE_NAMES.

    Returns:
        np.ndarray of shape (n,) with values clipped to [0, 100].
    """
    preds = model.predict(features.astype(np.float32))
    return np.clip(preds, 0.0, 100.0)


def features_dict_to_matrix(row: Dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a (1, 3) numpy array for inference."""
    return np.array(
        [[row["skill_match"], row["experience_match"], row["similarity_score"]]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: RandomForestRegressor, path: "str | Path") -> None:
    """Persist *model* to *path* using joblib compression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path: "str | Path") -> RandomForestRegressor:
    """Load a model from *path* (joblib format)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Backward-compatible helper used by api_fastapi.py and run_phases_demo.py
# ---------------------------------------------------------------------------

def ensure_default_model(
    models_dir: "str | Path | None" = None,
) -> RandomForestRegressor:
    """
    Return the singleton score model, warm-loading if needed.

    Backward-compatible with the original ``ensure_default_model`` signature.
    Callers that previously called this on every request now get the already-
    loaded singleton with zero I/O overhead after the first call.
    """
    reg = warm_up(Path(models_dir) if models_dir else None)
    if reg.score_model is not None:
        return reg.score_model

    # Last-resort: train in-process (should not normally happen)
    logger.warning("score_model unavailable after warm-up — training in-process.")
    m = train_model()
    reg.score_model = m
    return m
