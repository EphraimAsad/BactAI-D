# engine/schema.py
# ------------------------------------------------------------
# Core schema + Extended schema support
# ------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Any, Tuple
import json
import os

# ============================
# CORE SCHEMA DEFINITIONS
# ============================

POS_NEG_VAR = ["Positive", "Negative", "Variable"]
UNKNOWN = "Unknown"
MULTI_SEPARATOR = ";"


ENUMS = {
    "Gram Stain": ["Positive", "Negative", "Variable"],
    "Shape": ["Cocci", "Rods", "Bacilli", "Spiral", "Short Rods"],
    "Haemolysis Type": ["None", "Beta", "Gamma", "Alpha"],
}


SCHEMA: Dict[str, Dict[str, Any]] = {
    "Genus": {"type": "text", "required": True},
    "Species": {"type": "text", "required": False},

    "Gram Stain": {"type": "enum", "allowed": ENUMS["Gram Stain"]},
    "Shape": {"type": "enum", "allowed": ENUMS["Shape"]},
    "Colony Morphology": {"type": "multienum", "separator": MULTI_SEPARATOR},
    "Haemolysis": {"type": "enum", "allowed": POS_NEG_VAR},
    "Haemolysis Type": {"type": "multienum", "separator": MULTI_SEPARATOR, "allowed": ENUMS["Haemolysis Type"]},

    "Motility": {"type": "enum", "allowed": POS_NEG_VAR},
    "Capsule": {"type": "enum", "allowed": POS_NEG_VAR},
    "Spore Formation": {"type": "enum", "allowed": POS_NEG_VAR},

    "Growth Temperature": {"type": "range", "format": "low//high", "units": "°C"},
    "Oxygen Requirement": {"type": "text"},
    "Media Grown On": {"type": "multienum", "separator": MULTI_SEPARATOR},

    "Catalase": {"type": "enum", "allowed": POS_NEG_VAR},
    "Oxidase": {"type": "enum", "allowed": POS_NEG_VAR},
    "Indole": {"type": "enum", "allowed": POS_NEG_VAR},
    "Urease": {"type": "enum", "allowed": POS_NEG_VAR},
    "Citrate": {"type": "enum", "allowed": POS_NEG_VAR},
    "Methyl Red": {"type": "enum", "allowed": POS_NEG_VAR},
    "VP": {"type": "enum", "allowed": POS_NEG_VAR},
    "H2S": {"type": "enum", "allowed": POS_NEG_VAR},
    "DNase": {"type": "enum", "allowed": POS_NEG_VAR},
    "ONPG": {"type": "enum", "allowed": POS_NEG_VAR},
    "Coagulase": {"type": "enum", "allowed": POS_NEG_VAR},
    "Lipase Test": {"type": "enum", "allowed": POS_NEG_VAR},
    "Nitrate Reduction": {"type": "enum", "allowed": POS_NEG_VAR},

    "NaCl Tolerant (>=6%)": {"type": "enum", "allowed": POS_NEG_VAR},

    "Lysine Decarboxylase": {"type": "enum", "allowed": POS_NEG_VAR},
    "Ornitihine Decarboxylase": {"type": "enum", "allowed": POS_NEG_VAR},
    "Arginine dihydrolase": {"type": "enum", "allowed": POS_NEG_VAR},

    "Gelatin Hydrolysis": {"type": "enum", "allowed": POS_NEG_VAR},
    "Esculin Hydrolysis": {"type": "enum", "allowed": POS_NEG_VAR},

    "Glucose Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},
    "Lactose Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},
    "Sucrose Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},
    "Mannitol Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},
    "Sorbitol Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},
    "Maltose Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},
    "Xylose Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},
    "Rhamnose Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},
    "Arabinose Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},
    "Raffinose Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},
    "Trehalose Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},
    "Inositol Fermentation": {"type": "enum", "allowed": POS_NEG_VAR},

    "Extra Notes": {"type": "text"},
}


FIELDS_ORDER: List[str] = list(SCHEMA.keys())

MULTI_FIELDS: List[str] = [
    f for f, meta in SCHEMA.items() if meta.get("type") == "multienum"
]

PNV_FIELDS: List[str] = [
    f for f, meta in SCHEMA.items()
    if meta.get("type") == "enum" and meta.get("allowed") == POS_NEG_VAR
]

# ============================================================
# EXTENDED SCHEMA SUPPORT (needed for Stage 10C)
# ============================================================

def get_core_fields() -> List[str]:
    """Return the exact core schema fields (columns in DB)."""
    return list(SCHEMA.keys())


def load_extended_schema(path: str = "data/extended_schema.json") -> Dict[str, Any]:
    """Load extended schema from JSON; always returns a dict."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_extended_schema(schema: Dict[str, Any], path: str = "data/extended_schema.json") -> None:
    """Save updated extended schema."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


# ============================================================
# NORMALIZATION / VALIDATION (your existing logic preserved)
# ============================================================

def normalize_value(field: str, value: str) -> str:
    if value is None or str(value).strip() == "":
        return UNKNOWN
    v = str(value).strip()

    if v.lower() == "unknown":
        return UNKNOWN

    meta = SCHEMA.get(field, {})
    ftype = meta.get("type")

    if ftype == "enum":
        allowed = meta.get("allowed", [])
        for a in allowed:
            if v.lower() == a.lower():
                return a
        if v.lower() in ["+", "positive", "pos"]:
            return "Positive"
        if v.lower() in ["-", "negative", "neg"]:
            return "Negative"
        if v.lower() in ["variable", "var", "v"]:
            return "Variable"
        return v

    if ftype == "multienum":
        parts = [p.strip() for p in v.split(MULTI_SEPARATOR) if p.strip()]
        allowed = meta.get("allowed")
        normed = []
        for p in parts:
            if allowed:
                hit = next((a for a in allowed if a.lower() == p.lower()), None)
                normed.append(hit if hit else p)
            else:
                normed.append(p)
        return "; ".join(normed) if normed else UNKNOWN

    if ftype == "range":
        return v.replace(" ", "")

    return v


def validate_record(rec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues = []
    for field in FIELDS_ORDER:
        if field not in rec:
            continue
        val = rec[field]
        meta = SCHEMA[field]

        if meta["type"] == "enum":
            allowed = meta.get("allowed", [])
            if str(val) not in allowed + [UNKNOWN]:
                issues.append(f"{field}: '{val}' invalid")

        elif meta["type"] == "multienum":
            if val == UNKNOWN:
                continue
            parts = [p.strip() for p in val.split(MULTI_SEPARATOR)]
            allowed = meta.get("allowed")
            if allowed:
                bad = [p for p in parts if p not in allowed]
                if bad:
                    issues.append(f"{field}: invalid values {bad}")

        elif meta["type"] == "range":
            if val == UNKNOWN:
                continue
            if "//" not in str(val):
                issues.append(f"{field}: malformed range '{val}'")
    return (len(issues) == 0), issues


def empty_record() -> Dict[str, str]:
    rec = {}
    for f in SCHEMA.keys():
        rec[f] = "" if f in ("Genus", "Species") else UNKNOWN
    return rec