# engine/parser_ext.py
# ======================================================================
# Extended test parser — Stage 12C-fix4
#
# GOAL:
#   • Explicit-only parsing
#   • ML-safe
#   • Deterministic
#   • No inference
#   • Schema-backed fallback for enum_PNV
# ======================================================================

from __future__ import annotations
import os, re, json
from typing import Dict, Any

EXTENDED_SCHEMA_PATH = os.path.join("data", "extended_schema.json")

UNKNOWN = "Unknown"

# ======================================================================
# Fields NOT parsed here
# ======================================================================
CORE_FIELDS = {
    "Genus","Species",
    "Gram Stain","Shape","Colony Morphology",
    "Haemolysis","Motility","Capsule","Spore Formation",
    "Growth Temperature","Oxygen Requirement","Media Grown On",
    "Catalase","Oxidase","Indole","Urease","Citrate","Methyl Red","VP",
    "H2S","DNase","ONPG","Coagulase","Lipase Test","Nitrate Reduction",
    "Lysine Decarboxylase","Arginine dihydrolase",
    "Gelatin Hydrolysis","Esculin Hydrolysis",
    "Glucose Fermentation","Lactose Fermentation","Sucrose Fermentation",
    "Mannitol Fermentation","Sorbitol Fermentation","Maltose Fermentation",
    "Xylose Fermentation","Rhamnose Fermentation","Arabinose Fermentation",
    "Raffinose Fermentation","Trehalose Fermentation","Inositol Fermentation",
}

# ======================================================================
# Helpers
# ======================================================================

def _clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("°", "").replace("º", "").replace("₂", "2")
    return " ".join(t.split())

def _set_if_stronger(parsed: Dict[str,str], field: str, value: str):
    if not value:
        return
    if field not in parsed or parsed[field] == UNKNOWN:
        parsed[field] = value

def _parse_pnv_after_anchor(text: str, parsed: Dict[str,str], field: str, anchor: str):
    m = re.search(
        rf"\b{re.escape(anchor)}\b\s*(positive|negative|variable|unknown)",
        text,
        re.IGNORECASE,
    )
    if m:
        _set_if_stronger(parsed, field, m.group(1).capitalize())

def _load_extended_schema(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

# ======================================================================
# 1. Gram Stain Variable (explicit-only)
# ======================================================================

def _parse_gram_variable(text: str, parsed: Dict[str,str]):
    t = text.lower()
    if (
        re.search(r"\bgram[- ]variable\b", t) or
        re.search(r"\bgram stain variable\b", t) or
        re.search(r"\bvariable gram stain\b", t)
    ):
        _set_if_stronger(parsed, "Gram Stain", "Variable")

# ======================================================================
# 2. Shape (yeast phrasing fix)
# ======================================================================

def _parse_shape_yeast(text: str, parsed: Dict[str,str]):
    if re.search(r"\byeast cells?\b", text.lower()):
        _set_if_stronger(parsed, "Shape", "Yeast")

# ======================================================================
# 3. Capsule (explicit Variable only)
# ======================================================================

def _parse_capsule_variable(text: str, parsed: Dict[str,str]):
    t = text.lower()
    patterns = [
        r"\bcapsule\s*[:\-]?\s*variable\b",
        r"\bcapsule-variable\b",
        r"\bvariable\s+capsule\b",
    ]
    if any(re.search(p, t) for p in patterns):
        _set_if_stronger(parsed, "Capsule", "Variable")

# ======================================================================
# 4. Gas Production
# ======================================================================

def _parse_gas_production(text: str, parsed: Dict[str,str]):
    t = text.lower()
    POS = [
        "produces gas","gas produced","with gas",
        "gas production positive","gas producer",
        "production of gas","ferments glucose with gas",
    ]
    NEG = [
        "does not produce gas","no gas",
        "absence of gas","gas production negative",
    ]
    if any(p in t for p in POS):
        _set_if_stronger(parsed,"Gas Production","Positive")
    elif any(n in t for n in NEG):
        _set_if_stronger(parsed,"Gas Production","Negative")

# ======================================================================
# 5. Motility Type (explicit)
# ======================================================================

MOTILITY_TYPES = {
    "Peritrichous","Monotrichous","Polytrichous","Polar",
    "Swarming","Tumbling","Gliding","Corkscrew","Axial",
}

def _parse_motility_type(text: str, parsed: Dict[str,str]):
    t = text.lower()

    mneg = re.search(r"\bmotility type\b\s*[:\-]?\s*(negative|none)\b", t)
    if mneg:
        _set_if_stronger(parsed, "Motility Type", mneg.group(1).capitalize())
        return

    m = re.search(r"\bmotility type\b\s*[:\-]?\s*([a-z]+)", t)
    if m:
        val = m.group(1).capitalize()
        if val in MOTILITY_TYPES:
            _set_if_stronger(parsed, "Motility Type", val)
            return

    for mt in MOTILITY_TYPES:
        if re.search(rf"\b{mt.lower()}\b", t):
            _set_if_stronger(parsed, "Motility Type", mt)
            return

# ======================================================================
# 6. Pigment (EXPLICIT + SCIENTIFIC TERMS ONLY)
# ======================================================================

SCIENTIFIC_PIGMENTS = {
    "Pyocyanin","Pyoverdine","Pyovacin","Bioluminescent"
}

COLOUR_PIGMENTS = {
    "green","yellow","pink","red","orange","brown","black","violet","cream"
}

def _parse_pigment(text: str, parsed: Dict[str,str]):
    t = text.lower()

    # Joint negative phrase
    if re.search(r"\bno pigmentation or odou?r\b", t):
        _set_if_stronger(parsed, "Pigment", "None")
        _set_if_stronger(parsed, "Odor", "None")
        return

    has_anchor = re.search(r"\b(pigment|pigmentation)\b", t)
    found = set()

    # Scientific pigments (allowed without anchor)
    for sp in SCIENTIFIC_PIGMENTS:
        if re.search(rf"\b{sp.lower()}\b", t):
            found.add(sp)

    # Colour pigments ONLY if pigment anchor exists
    if has_anchor:
        for cp in COLOUR_PIGMENTS:
            if re.search(rf"\b{cp}\b", t):
                found.add(cp.capitalize())

    if re.search(r"\bno pigmentation\b|\bpigment none\b", t):
        _set_if_stronger(parsed, "Pigment", "None")
    elif found:
        _set_if_stronger(parsed, "Pigment", "; ".join(sorted(found)))

# ======================================================================
# 7. Colony Pattern (explicit only)
# ======================================================================

COLONY_PATTERNS = {
    "Mucoid","Smooth","Rough","Filamentous",
    "Spreading","Swarming","Sticky","Irregular",
    "Ground-glass","Molar-tooth","Dry","Chalky","Corroding",
}

def _parse_colony_pattern(text: str, parsed: Dict[str,str]):
    t = text.lower()
    if not re.search(r"\bcolony pattern\b", t):
        return
    m = re.search(r"\bcolony pattern\b\s*[:\-]?\s*([a-z\-]+)", t)
    if m:
        val = m.group(1).capitalize()
        if val in COLONY_PATTERNS:
            _set_if_stronger(parsed, "Colony Pattern", val)

# ======================================================================
# 8. Odor (explicit anchor-based)
# ======================================================================

def _parse_odor(text: str, parsed: Dict[str,str]):
    t = text.lower()
    m = re.search(r"\b(odor|odour|smell)\b\s*[:\-]?\s*([a-z; ]+)", t)
    if not m:
        return
    vals = [v.strip().capitalize() for v in m.group(2).split(";") if v.strip()]
    if vals:
        _set_if_stronger(parsed, "Odor", "; ".join(vals))

# ======================================================================
# 9. TSI Pattern
# ======================================================================

def _parse_tsi(text: str, parsed: Dict[str,str]):
    t = text.upper()
    if "TSI" in t and "UNKNOWN" in t:
        _set_if_stronger(parsed, "TSI Pattern", "Unknown")
        return
    m = re.search(r"\b([KA]/[KA])(\s*\+\s*H2S)?\b", t)
    if m:
        base = m.group(1)
        _set_if_stronger(parsed, "TSI Pattern", f"{base}+H2S" if m.group(2) else base)

# ======================================================================
# 10. NaCl Tolerant (>=6%)
# ======================================================================

def _parse_nacl(text: str, parsed: Dict[str,str]):
    m = re.search(
        r"NaCl\s*Tolerant\s*\(>=\s*6%\)\s*(positive|negative|variable|unknown)",
        text,
        re.IGNORECASE,
    )
    if m:
        _set_if_stronger(parsed, "NaCl Tolerant (>=6%)", m.group(1).capitalize())
        return
    _parse_pnv_after_anchor(text, parsed, "NaCl Tolerant (>=6%)", "NaCl Tolerant")

# ======================================================================
# 11. Haemolysis Type
# ======================================================================

def _parse_haemolysis_type(text: str, parsed: Dict[str,str]):
    m = re.search(
        r"\bhaemolysis type\b\s*[:\-]?\s*(alpha|beta|gamma|none)",
        text,
        re.IGNORECASE,
    )
    if m:
        _set_if_stronger(parsed, "Haemolysis Type", m.group(1).capitalize())

# ======================================================================
# 12. Ornithine Decarboxylase (both spellings)
# ======================================================================

def _parse_ornithine_dec(text: str, parsed: Dict[str,str]):
    _parse_pnv_after_anchor(text, parsed, "Ornithine Decarboxylase", "Ornithine Decarboxylase")
    _parse_pnv_after_anchor(text, parsed, "Ornitihine Decarboxylase", "Ornitihine Decarboxylase")
    if "Ornitihine Decarboxylase" in parsed and "Ornithine Decarboxylase" not in parsed:
        _set_if_stronger(parsed, "Ornithine Decarboxylase", parsed["Ornitihine Decarboxylase"])

# ======================================================================
# 13. Schema-driven enum_PNV fallback (SAFE)
# ======================================================================

def _parse_schema_enum_pnv(text: str, parsed: Dict[str,str]):
    schema = _load_extended_schema(EXTENDED_SCHEMA_PATH)
    t = text.lower()
    for field, meta in schema.items():
        if field in CORE_FIELDS or field in parsed:
            continue
        if meta.get("value_type") != "enum_PNV":
            continue
        aliases = meta.get("aliases", [])
        for name in [field] + aliases:
            m = re.search(
                rf"\b{re.escape(name.lower())}\b\s*(positive|negative|variable|unknown)",
                t,
            )
            if m:
                _set_if_stronger(parsed, field, m.group(1).capitalize())
                break

# ======================================================================
# MAIN
# ======================================================================

def parse_text_extended(text: str) -> Dict[str,Any]:
    orig = text or ""
    if not orig.strip():
        return {"parsed_fields": {}, "source": "extended_parser", "raw": orig}

    cleaned = _clean_text(orig)
    parsed: Dict[str,str] = {}

    _parse_gram_variable(cleaned, parsed)
    _parse_shape_yeast(cleaned, parsed)
    _parse_capsule_variable(cleaned, parsed)
    _parse_gas_production(cleaned, parsed)
    _parse_motility_type(cleaned, parsed)
    _parse_pigment(cleaned, parsed)
    _parse_colony_pattern(cleaned, parsed)
    _parse_odor(cleaned, parsed)
    _parse_tsi(cleaned, parsed)
    _parse_nacl(cleaned, parsed)
    _parse_haemolysis_type(cleaned, parsed)
    _parse_ornithine_dec(cleaned, parsed)
    _parse_schema_enum_pnv(cleaned, parsed)

    return {
        "parsed_fields": parsed,
        "source": "extended_parser",
        "raw": orig,
    }