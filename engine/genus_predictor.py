# engine/genus_predictor.py
"""
Genus-level ML prediction using the XGBoost model trained in Stage 12D.

This module loads:
    models/genus_xgb.json
    models/genus_xgb_meta.json

And exposes:
    predict_genus_from_fused(fused_fields)

Which returns a list of tuples:
    [
        (genus_name, probability_float, confidence_label),
        ...
    ]

Where confidence_label is one of:
    - "Excellent Identification"   (>= 0.90)
    - "Good Identification"        (>= 0.80)
    - "Acceptable Identification"  (>= 0.65)
    - "Low Discrimination"         (< 0.65)
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import xgboost as xgb

from .features import extract_feature_vector


# Paths
_MODEL_PATH = "models/genus_xgb.json"
_META_PATH = "models/genus_xgb_meta.json"


# ----------------------------------------------------------------------
# Lazy load model + metadata — only loads once globally
# ----------------------------------------------------------------------

_MODEL = None
_META = None
_IDX_TO_GENUS = None
_NUM_FEATURES = None
_NUM_CLASSES = None


def _lazy_load():
    """Load model and metadata only once."""
    global _MODEL, _META, _IDX_TO_GENUS, _NUM_FEATURES, _NUM_CLASSES

    if _MODEL is not None:
        return

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(f"Genus model not found at '{_MODEL_PATH}'.")

    if not os.path.exists(_META_PATH):
        raise FileNotFoundError(f"Genus meta file not found at '{_META_PATH}'.")

    # Load model
    _MODEL = xgb.Booster()
    _MODEL.load_model(_MODEL_PATH)

    # Load metadata
    with open(_META_PATH, "r", encoding="utf-8") as f:
        _META = json.load(f)

    _IDX_TO_GENUS = {int(k): v for k, v in _META["idx_to_genus"].items()}
    _NUM_FEATURES = _META["n_features"]
    _NUM_CLASSES = _META["num_classes"]


# ----------------------------------------------------------------------
# Confidence label assignment
# ----------------------------------------------------------------------

def _confidence_band(p: float) -> str:
    if p >= 0.90:
        return "Excellent Identification"
    if p >= 0.80:
        return "Good Identification"
    if p >= 0.65:
        return "Acceptable Identification"
    return "Low Discrimination"


# ----------------------------------------------------------------------
# Public prediction function
# ----------------------------------------------------------------------

def predict_genus_from_fused(
    fused_fields: Dict[str, Any],
    top_k: int = 10
) -> List[Tuple[str, float, str]]:
    """
    Predict genus from fused fields using the trained XGBoost model.

    Returns top_k results sorted by probability:
        [(genus_name, probability_float, confidence_label), ...]
    """
    _lazy_load()

    # Build feature vector
    vec = extract_feature_vector(fused_fields)
    if vec.shape[0] != _NUM_FEATURES:
        # Defensive: mismatch in schema → pad or trim
        fixed = np.zeros(_NUM_FEATURES, dtype=float)
        m = min(len(vec), _NUM_FEATURES)
        fixed[:m] = vec[:m]
        vec = fixed

    dmat = xgb.DMatrix(vec.reshape(1, -1))
    probs = _MODEL.predict(dmat)[0]  # shape: (num_classes,)

    # Build list of (genus, prob, band)
    results = []
    for idx, p in enumerate(probs):
        genus = _IDX_TO_GENUS.get(idx, f"Class_{idx}")
        results.append((genus, float(p), _confidence_band(float(p))))

    # Sort by probability, descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]
