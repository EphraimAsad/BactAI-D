# engine/features.py
import json
import numpy as np
import re
from typing import Dict, List, Any

# ------------------------------------------------------------------
# Load schema once
# ------------------------------------------------------------------

_FEATURE_SCHEMA_PATH = "data/feature_schema.json"

with open(_FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
    SCHEMA = json.load(f)

FEATURES = SCHEMA["features"]

# ------------------------------------------------------------------
# Helper mappings
# ------------------------------------------------------------------

PNV_MAP = {
    "positive": 1.0,
    "negative": -1.0,
    "variable": 0.5,
    "unknown": 0.0,
    None: 0.0,
}

SHAPE_MAP = {
    "cocci": 1.0,
    "rods": 2.0,
    "short rods": 2.5,
    "spiral": 3.0,
    "yeast": 4.0,
    "variable": 0.5,
    "unknown": 0.0,
}

OXYGEN_MAP = {
    "aerobic": 1.0,
    "anaerobic": 2.0,
    "facultative anaerobe": 3.0,
    "microaerophilic": 4.0,
    "capnophilic": 5.0,
    "unknown": 0.0,
}

# CATEGORY LABELSETS — deterministic, fixed, ML-friendly
CATEGORY_MAPS = {
    "Motility Type": [
        "none", "tumbling", "peritrichous", "polar", "monotrichous",
        "lophotrichous", "amphitrichous", "axial", "gliding"
    ],
    "Pigment": [
        "none", "pyocyanin", "pyoverdine", "green", "yellow", "pink",
        "red", "orange", "brown", "black", "violet", "cream"
    ],
    "Odor": [
        "none", "grape", "fruity", "earthy", "musty", "putrid",
        "buttery", "yeasty", "medicinal", "fishy", "almond", "burnt", "mousy"
    ],
    "Colony Pattern": [
        "none", "mucoid", "rough", "smooth", "filamentous",
        "spreading", "chalky", "corroding", "swarming",
        "sticky", "ground-glass", "molar-tooth"
    ],
    "TSI Pattern": [
        "unknown", "a/a", "k/a", "k/k",
        "k/a+h2s", "a/a+gas"
    ],
}

# Make fast lookup: label → integer code
CATEGORY_ENCODERS = {
    field: {lab: idx for idx, lab in enumerate(labels)}
    for field, labels in CATEGORY_MAPS.items()
}

# Temperature flags: a direct binary interpretation
TEMP_FLAGS = {"4c", "25c", "30c", "37c", "42c"}

# ------------------------------------------------------------------
# Normalisation helpers
# ------------------------------------------------------------------

def _norm(x: Any) -> str:
    if not x:
        return "unknown"
    return str(x).strip().lower()


def _map_pnv(x: Any) -> float:
    return PNV_MAP.get(_norm(x), 0.0)


def _map_shape(x: Any) -> float:
    return SHAPE_MAP.get(_norm(x), 0.0)


def _map_oxygen(x: Any) -> float:
    return OXYGEN_MAP.get(_norm(x), 0.0)


def _extract_temperature_flags(value: str):
    """
    Convert things like "25//37" → {"25c":1, "37c":1}
    """
    flags = {k: 0.0 for k in TEMP_FLAGS}
    if not value:
        return flags

    s = value.lower()
    nums = re.findall(r"\b(\d{1,2})\s*c?\b", s)

    for n in nums:
        key = f"{n}c"
        if key in flags:
            flags[key] = 1.0

    return flags


def _growth_minmax(v: Any):
    """Convert '30//37' → (30,37)."""
    if not v:
        return (0.0, 0.0)
    if not isinstance(v, str):
        v = str(v)

    m = re.match(r"^\s*(\d+)\s*//\s*(\d+)\s*$", v)
    if not m:
        return (0.0, 0.0)
    return (float(m.group(1)), float(m.group(2)))


def _media_flag(media_field: Any, medium: str) -> float:
    if not media_field:
        return 0.0
    mf = str(media_field).lower()
    return 1.0 if medium.lower() in mf else 0.0


# ------------------------------------------------------------------
# CATEGORY mapping helper
# ------------------------------------------------------------------

def _map_category(field: str, value: Any) -> float:
    """
    Deterministic integer encoding.
    Unknown → 0 (first element)
    Multi-values like "yellow; orange" → choose first matching token.
    """
    labels = CATEGORY_MAPS.get(field)
    if not labels:
        return 0.0  # should not happen

    enc = CATEGORY_ENCODERS[field]
    s = _norm(value)

    # Multi-list: pick first match
    parts = [p.strip() for p in re.split(r"[;/,]", s) if p.strip()]

    for p in parts:
        if p in enc:
            return float(enc[p])

    # No match → return index for "none" or "unknown"
    fallback = "none" if "none" in enc else "unknown"
    return float(enc.get(fallback, 0))


# ------------------------------------------------------------------
# Main feature extractor
# ------------------------------------------------------------------

def extract_feature_vector(fused_fields: Dict[str, Any]) -> np.ndarray:
    """
    Convert fused fields into a fixed-length ML-ready numeric vector.
    ORDER must match feature_schema.json exactly.
    """
    vec: List[float] = []

    growth_temp = fused_fields.get("Growth Temperature")
    temp_flags = _extract_temperature_flags(growth_temp)

    for f in FEATURES:
        name = f["name"]
        kind = f["kind"]

        value = fused_fields.get(name, "Unknown")
        norm = _norm(value)

        # ---------------------------
        # pnv
        # ---------------------------
        if kind == "pnv":
            vec.append(_map_pnv(norm))

        # ---------------------------
        # shape
        # ---------------------------
        elif kind == "shape":
            vec.append(_map_shape(norm))

        # ---------------------------
        # oxygen requirement
        # ---------------------------
        elif kind == "oxygen":
            vec.append(_map_oxygen(norm))

        # ---------------------------
        # category → integer encoding
        # ---------------------------
        elif kind == "category":
            vec.append(_map_category(name, value))

        # ---------------------------
        # binary flag (temperatures)
        # ---------------------------
        elif kind == "binary":
            key = name.lower().replace("temperature_", "")
            vec.append(temp_flags.get(key, 0.0))

        # ---------------------------
        # media flag
        # ---------------------------
        elif kind == "media_flag":
            medium = name.replace("Growth", "").strip()
            media_field = fused_fields.get("Media Grown On")
            vec.append(_media_flag(media_field, medium))

        # ---------------------------
        # numeric_from_growth_temp (legacy support)
        # ---------------------------
        elif kind == "numeric_from_growth_temp":
            lo, hi = _growth_minmax(growth_temp)
            if "min" in name.lower():
                vec.append(lo)
            elif "max" in name.lower():
                vec.append(hi)
            else:
                vec.append(0.0)

        # ---------------------------
        # unknown kind
        # ---------------------------
        else:
            vec.append(0.0)

    return np.array(vec, dtype=float)