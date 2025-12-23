# engine/train_genus_model.py
"""
Train a genus-level classifier (XGBoost) from gold tests.

Pipeline:
  • Load gold_tests.json
  • Extract genus (first token of organism name)
  • Convert expected_fields → feature vector (via engine.features.extract_feature_vector)
  • Train an XGBoost multi-class classifier
  • Save:
        models/genus_xgb.json
        models/genus_xgb_meta.json

Compatible with FEATURE SCHEMA v2 (category, binary temperature flags, pigment, odor, colony pattern, TSI, etc.)
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import xgboost as xgb

from .features import extract_feature_vector, FEATURES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GOLD_TESTS_PATH = "training/gold_tests.json"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "genus_xgb.json")
META_PATH = os.path.join(MODEL_DIR, "genus_xgb_meta.json")


# ---------------------------------------------------------------------------
# Load gold tests
# ---------------------------------------------------------------------------

def _load_gold_tests(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing gold test file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("gold_tests.json must contain a list.")

    return data


# ---------------------------------------------------------------------------
# Extract genus & expected fields
# ---------------------------------------------------------------------------

def _extract_genus(sample: Dict[str, Any]) -> str | None:
    """
    Extract genus from:
        name / Name / organism / Organism
    (genus = first token before space)
    """
    for key in ("name", "Name", "organism", "Organism"):
        if key in sample and sample[key]:
            val = str(sample[key]).strip()
            if val:
                return val.split()[0]
    return None


def _extract_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract expected field dict from any of:
       fields / expected_fields / schema / expected
    """
    for key in ("fields", "expected_fields", "schema", "expected"):
        if key in sample and isinstance(sample[key], dict):
            return sample[key]
    return {}


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_dataset(samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Convert gold tests into:
       X  → feature matrix
       y  → integer labels
       genus_to_idx → mapping
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    genus_to_idx: Dict[str, int] = {}

    for sample in samples:
        genus = _extract_genus(sample)
        if not genus:
            continue

        fields = _extract_fields(sample)
        if not fields:
            continue

        # Generate ML feature vector (schema v2)
        vec = extract_feature_vector(fields)

        if genus not in genus_to_idx:
            genus_to_idx[genus] = len(genus_to_idx)

        X_list.append(vec)
        y_list.append(genus_to_idx[genus])

    if not X_list:
        raise ValueError("No usable gold tests found.")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)

    return X, y, genus_to_idx


# ---------------------------------------------------------------------------
# Train XGBoost model
# ---------------------------------------------------------------------------

def _train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    seed: int = 42
) -> Tuple[xgb.Booster, Dict[str, float]]:
    """
    Train a multi-class XGBoost classifier.
    80/20 split.
    """

    n = X.shape[0]
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    split = int(0.8 * n)
    train_idx = indices[:split]
    valid_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "max_depth": 6,        # Higher depth since schema v2 more complex
        "eta": 0.08,           # Slightly slower learning
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "seed": seed,
    }

    evals = [(dtrain, "train"), (dvalid, "valid")]

    model = xgb.train(
        params,
        dtrain,
        evals=evals,
        num_boost_round=500,      # More rounds since more features
        early_stopping_rounds=40,  # Allow more patience for complex space
        verbose_eval=50,
    )

    # Accuracy evaluation
    train_acc = float(
        (np.argmax(model.predict(dtrain), axis=1) == y_train).mean()
    )
    valid_acc = float(
        (np.argmax(model.predict(dvalid), axis=1) == y_valid).mean()
    )

    return model, {
        "train_accuracy": train_acc,
        "valid_accuracy": valid_acc,
        "best_iteration": int(model.best_iteration),
    }


def _ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Public entry for UI
# ---------------------------------------------------------------------------

def train_genus_model() -> Dict[str, Any]:
    try:
        print(f"Loading gold tests → {GOLD_TESTS_PATH}")
        samples = _load_gold_tests(GOLD_TESTS_PATH)

        print("Building ML dataset...")
        X, y, genus_to_idx = _build_dataset(samples)

        num_classes = len(genus_to_idx)
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Classes (genera):   {num_classes}")
        print(f"Samples:            {X.shape[0]}")

        print("Training XGBoost (schema v2)...")
        model, metrics = _train_xgboost(X, y, num_classes)

        print("Training complete.")
        print(f"Train accuracy: {metrics['train_accuracy']:.3f}")
        print(f"Valid accuracy: {metrics['valid_accuracy']:.3f}")

        _ensure_model_dir()
        model.save_model(MODEL_PATH)

        idx_to_genus = {idx: genus for genus, idx in genus_to_idx.items()}

        meta = {
            "genus_to_idx": genus_to_idx,
            "idx_to_genus": idx_to_genus,
            "n_features": int(X.shape[1]),
            "num_classes": int(num_classes),
            "metrics": metrics,
            "feature_schema_path": "data/feature_schema.json",
            "feature_names": [f["name"] for f in FEATURES],
        }

        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return {
            "ok": True,
            "message": "Genus XGBoost model (schema v2) trained successfully.",
            "stats": {
                "num_raw_samples": len(samples),
                "num_usable_samples": int(X.shape[0]),
                "feature_dim": int(X.shape[1]),
                "num_classes": int(num_classes),
            },
            "metrics": metrics,
            "paths": {"model_path": MODEL_PATH, "meta_path": META_PATH},
            "genus_examples": sorted(genus_to_idx.keys())[:20],
        }

    except Exception as e:
        return {
            "ok": False,
            "message": f"Training error: {type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    print(json.dumps(train_genus_model(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()