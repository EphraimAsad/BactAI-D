# engine/parser_rules.py
# ------------------------------------------------------------
# Rule-based core parser for microbiology descriptions.
#
# Stage 11F (Option A ranges + fixes) + 11H + 11I + 11J + 11L + 11M
# + NaCl + haemolysis symbol support + colony morphology tweaks.
#
# - Always store Growth Temperature as "low//high"
#   • single: 37 → "37//37"
#   • any two temps in text: min//max
#   • ranges like "30–37 °C", "grows between 30 and 37 °C" → "30//37"
#
# - DNase robust parsing (DNase test / activity / production)
# - Non–spore-forming → Spore Formation = Negative (with early return)
# - "non-H2S producing" → H2S = Negative
# - Aerobic / Anaerobic including “aerobically / anaerobically”
#
# - NaCl tolerance phrases improved (>= 6% rule)
#     • explicit positives require a growth/tolerance verb + % ≥ 6
#     • explicit negatives ("no growth in NaCl", "does not grow in 7% NaCl",
#       "NaCl sensitive", "not NaCl tolerant") override positives
#     • ambiguous "in 6.5% NaCl" alone no longer auto-Positive
#
# - Colony morphology extraction, including:
#     • "colonies are yellow, mucoid"
#     • "colonies dry, white and irregular on nutrient agar"
#     • "forming smooth, yellow-pigmented, opaque colonies"
#     • "grey colonies", "large grey colonies" etc.
#
# - Sugars:
#     • "<sugar> positive/negative"
#     • "<sugar> is positive/negative"
#     • "<sugar> fermenter" / "non-<sugar> fermenter"
#     • "ferments X, Y but not Z"
#     • grouped "does not ferment lactose and sucrose"
#       (without nuking glucose in "but glucose positive")
#     • global "non-fermenter" → all sugars Negative (Unknown-only)
#     • "asaccharolytic" → all sugars Negative (Unknown-only)
#     • "all other sugars negative" → all remaining sugars Negative
#       (Unknown-only; no hard rewrite)
#
# - Core tests:
#     • "<kw> positive/negative"
#     • "positive for <kw>"
#     • "<kw> is positive/negative"
#     • "<kw> reaction is positive/negative"
#     • "<kw> reaction positive/negative"
#     • "<kw> test reaction is positive/negative"
#     • "ONPG is negative" handled via core patterns
#     • "H2S production is positive/negative"
#     • "MR and VP negative/positive" → both set
#     • grouped phrases like
#         "gelatin and esculin hydrolysis negative"
#         "lysine, ornithine and arginine negative"
#       → all mentioned tests / sugars set to the given value
#
# - Decarboxylases:
#     • "all decarboxylases negative/positive"
#       → Lysine / Ornithine / Arginine dihydrolase set accordingly
#       (Unknown-only; explicit values can override later)
#
# - Capsule / Motility:
#     • "capsule present"/"capsule is present" → Capsule Positive
#     • "capsule absent"/"capsule is absent"/"no capsule" → Capsule Negative
#     • "encapsulated" / "capsulated" → Capsule Positive
#     • "gliding/spreading/swarming motility" → Motility Positive
#
# - Gelatin / Esculin:
#     • "gelatin positive/negative" → Gelatin Hydrolysis
#     • "esculin positive/negative" → Esculin Hydrolysis
#
# - Shape:
#     • "coccobacilli / coccobacillus" → Shape = Short Rods
#     • (no 4F shape descriptor explosion; we keep existing logic)
#
# - Haemolysis:
#     • alpha/beta/gamma haemolysis & haemolytic
#     • now also supports α / β / γ symbols via normalisation
# ------------------------------------------------------------

from __future__ import annotations

import re
from typing import Dict, Any, List


UNKNOWN = "Unknown"

# ------------------------------------------------------------
# Core fields and sugar mapping
# ------------------------------------------------------------

# Sugar name → core DB column
SUGAR_FIELDS: Dict[str, str] = {
    "glucose": "Glucose Fermentation",
    "lactose": "Lactose Fermentation",
    "sucrose": "Sucrose Fermentation",
    "maltose": "Maltose Fermentation",
    "mannitol": "Mannitol Fermentation",
    "sorbitol": "Sorbitol Fermentation",
    "xylose": "Xylose Fermentation",
    "rhamnose": "Rhamnose Fermentation",
    "arabinose": "Arabinose Fermentation",
    "raffinose": "Raffinose Fermentation",
    "trehalose": "Trehalose Fermentation",
    "inositol": "Inositol Fermentation",
}

CORE_BOOL_FIELDS: Dict[str, List[str]] = {
    # field: [keywords to recognise the test name]
    "Catalase": ["catalase"],
    "Oxidase": ["oxidase"],
    "Indole": ["indole"],
    "Urease": ["urease"],
    "Citrate": ["citrate"],
    # MR: include "mr"
    "Methyl Red": ["methyl red", "mr test", "mr"],
    "VP": ["voges-proskauer", "vp test", "vp"],
    # H2S (includes H₂S → normalised to H2S in _clean_text)
    "H2S": ["h2s", "hydrogen sulfide"],
    # DNase: broaden patterns
    "DNase": [
        "dnase",
        "dnase test",
        "dnase activity",
        "dnase production",
        "dnaase",
        "dna hydrolysis",
    ],
    "ONPG": ["onpg"],
    "Coagulase": ["coagulase"],
    "Lipase Test": ["lipase"],
    "Nitrate Reduction": ["nitrate reduction", "nitrate"],
    "NaCl Tolerant (>=6%)": ["6% nacl", "7% nacl", "nacl tolerant"],
    # Decarboxylases (also match plain amino acid words)
    "Lysine Decarboxylase": ["lysine decarboxylase", "lysine decarb", "lysine"],
    "Ornitihine Decarboxylase": ["ornithine decarboxylase", "ornithine decarb", "ornithine"],
    "Arginine dihydrolase": ["arginine dihydrolase", "arginine decarboxylase", "arginine"],
    # Gelatin / Esculin
    "Gelatin Hydrolysis": ["gelatin hydrolysis", "gelatinase", "gelatin"],
    "Esculin Hydrolysis": ["esculin hydrolysis", "esculin"],
}

# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Normalise unicode oddities and collapse whitespace.
    Also:
      - strip degree symbols
      - normalise subscript ₂ → 2 for H₂S
      - normalise α/β/γ to alpha/beta/gamma for haemolysis patterns
    """
    if not text:
        return ""
    s = text.replace("°", "").replace("º", "")
    # normalise subscript 2 (H₂S → H2S)
    s = s.replace("₂", "2")

    # Greek letters for haemolysis and related descriptors
    s = (
        s.replace("α", "alpha")
         .replace("β", "beta")
         .replace("γ", "gamma")
    )

    # collapse whitespace
    return " ".join(s.split())


def _norm(s: str) -> str:
    return s.strip().lower()


def _set_if_stronger(parsed: Dict[str, str], field: str, value: str) -> None:
    """
    Write value to parsed[field] if:
      - field not present, or
      - we are replacing Unknown with a concrete value
    """
    if not value:
        return
    if field not in parsed or parsed[field] == UNKNOWN:
        parsed[field] = value


def _value_from_pnv_token(token: str) -> str | None:
    """
    Map a simple token to Positive / Negative / Variable.
    """
    seg = _norm(token)
    if seg in ["positive", "pos", "+"]:
        return "Positive"
    if seg in ["negative", "neg", "-"]:
        return "Negative"
    if seg in ["variable", "var", "v"]:
        return "Variable"
    return None


def _value_from_pnv_context(segment: str) -> str | None:
    """
    Interpret a phrase as Positive / Negative / Variable.

    Handles:
      - "positive"
      - "is positive"
      - "+", "neg", etc.
    """
    seg = _norm(segment)
    # direct token first
    val = _value_from_pnv_token(seg)
    if val:
        return val
    # "... is positive"
    m = re.search(r"\bis\s+(positive|negative|variable|pos|neg|\+|\-)\b", seg)
    if m:
        return _value_from_pnv_token(m.group(1))
    return None


# ------------------------------------------------------------
# Gram stain and shape
# ------------------------------------------------------------

def _parse_gram_and_shape(text_lc: str, parsed: Dict[str, str]) -> None:
    # Gram stain
    if "gram-positive" in text_lc or "gram positive" in text_lc:
        _set_if_stronger(parsed, "Gram Stain", "Positive")
    elif "gram-negative" in text_lc or "gram negative" in text_lc:
        _set_if_stronger(parsed, "Gram Stain", "Negative")
    elif "gram variable" in text_lc:
        _set_if_stronger(parsed, "Gram Stain", "Variable")

    # Shape
    # Prefer "short rods" / coccobacilli over generic rods
    if "short rods" in text_lc:
        _set_if_stronger(parsed, "Shape", "Short Rods")

    # NEW: coccobacilli → Short Rods
    if re.search(r"\bcoccobacill(?:us|i)\b", text_lc):
        _set_if_stronger(parsed, "Shape", "Short Rods")

    # Cocci and variants (diplococci, tetracocci, etc.)
    if re.search(r"\bcocci\b", text_lc):
        _set_if_stronger(parsed, "Shape", "Cocci")
    if re.search(r"\b(diplococci|tetracocci|streptococci|staphylococci)\b", text_lc):
        _set_if_stronger(parsed, "Shape", "Cocci")

    # Rods / bacilli
    if re.search(r"\brods?\b", text_lc) or "bacilli" in text_lc:
        _set_if_stronger(parsed, "Shape", "Rods")

    # Spiral
    if "spiral" in text_lc or "spirochete" in text_lc:
        _set_if_stronger(parsed, "Shape", "Spiral")


# ------------------------------------------------------------
# Haemolysis
# ------------------------------------------------------------

def _parse_haemolysis(text_lc: str, parsed: Dict[str, str]) -> None:
    """
    Handle haemolysis phrasing:
      - beta-haemolytic / beta hemolytic / beta-haemolysis / etc.
      - alpha- / gamma- / non-haemolytic
      - α / β / γ symbols are normalised to alpha/beta/gamma in _clean_text
    """
    # Beta
    if re.search(r"beta[- ]?(haemolytic|hemolytic|haemolysis|hemolysis)", text_lc):
        _set_if_stronger(parsed, "Haemolysis Type", "Beta")
        _set_if_stronger(parsed, "Haemolysis", "Positive")

    # Alpha
    if re.search(r"alpha[- ]?(haemolytic|hemolytic|haemolysis|hemolysis)", text_lc):
        _set_if_stronger(parsed, "Haemolysis Type", "Alpha")
        _set_if_stronger(parsed, "Haemolysis", "Positive")

    # Gamma / non-haemolytic
    if re.search(r"gamma[- ]?(haemolytic|hemolytic|haemolysis|hemolysis)", text_lc):
        _set_if_stronger(parsed, "Haemolysis Type", "Gamma")
        _set_if_stronger(parsed, "Haemolysis", "Negative")
    if (
        "non-haemolytic" in text_lc
        or "non hemolytic" in text_lc
        or "non-hemolytic" in text_lc
    ):
        _set_if_stronger(parsed, "Haemolysis Type", "None")
        _set_if_stronger(parsed, "Haemolysis", "Negative")

    # Variable phrasing
    if "variable haemolysis" in text_lc or "variable hemolysis" in text_lc:
        _set_if_stronger(parsed, "Haemolysis Type", "Variable")
        _set_if_stronger(parsed, "Haemolysis", "Variable")


# ------------------------------------------------------------
# Core enzyme / boolean tests
# ------------------------------------------------------------

def _parse_core_bool_tests(text_lc: str, parsed: Dict[str, str]) -> None:
    """
    For each test in CORE_BOOL_FIELDS, look for patterns like:
      - "catalase positive"
      - "positive for catalase"
      - "catalase is positive"
      - "indole reaction is negative"
      - "indole reaction negative"
      - "indole test reaction is positive"
    Plus:
      - NaCl tolerance with % values
      - Nitrate reduction text
      - H2S production / non-production
      - DNase coverage
      - gelatinase / gelatin → Gelatin Hydrolysis
      - esculin → Esculin Hydrolysis
      - grouped MR/VP: "MR and VP negative"
      - decarboxylase global phrases
      - generic grouped phrases
        "gelatin and esculin hydrolysis negative"
        "lysine, ornithine and arginine negative"
    """
    for field, keywords in CORE_BOOL_FIELDS.items():
        for kw in keywords:
            # 1) "... catalase positive"
            m1 = re.search(
                rf"{re.escape(kw)}[ \-]?"
                r"(positive|negative|variable|pos|neg|\+|\-)",
                text_lc,
            )
            if m1:
                val = _value_from_pnv_context(m1.group(1))
                if val:
                    _set_if_stronger(parsed, field, val)
                    break

            # 2) "positive for catalase"
            m2 = re.search(
                rf"(positive|negative|variable|pos|neg|\+|\-)\s+"
                rf"(for\s+)?{re.escape(kw)}",
                text_lc,
            )
            if m2:
                val = _value_from_pnv_context(m2.group(1))
                if val:
                    _set_if_stronger(parsed, field, val)
                    break

            # 3) "<kw> is positive"
            m3 = re.search(
                rf"{re.escape(kw)}\s+is\s+"
                r"(positive|negative|variable|pos|neg|\+|\-)",
                text_lc,
            )
            if m3:
                val = _value_from_pnv_token(m3.group(1))
                if val:
                    _set_if_stronger(parsed, field, val)
                    break

            # 4) "<kw> reaction is positive/negative"
            m4 = re.search(
                rf"{re.escape(kw)}\s+reaction\s+is\s+"
                r"(positive|negative|variable|pos|neg|\+|\-)",
                text_lc,
            )
            if m4:
                val = _value_from_pnv_token(m4.group(1))
                if val:
                    _set_if_stronger(parsed, field, val)
                    break

            # 5) "<kw> reaction positive/negative"
            m5 = re.search(
                rf"{re.escape(kw)}\s+reaction\s+"
                r"(positive|negative|variable|pos|neg|\+|\-)",
                text_lc,
            )
            if m5:
                val = _value_from_pnv_token(m5.group(1))
                if val:
                    _set_if_stronger(parsed, field, val)
                    break

            # 6) "<kw> test reaction is positive"
            m6 = re.search(
                rf"{re.escape(kw)}\s+test\s+reaction\s+is\s+"
                r"(positive|negative|variable|pos|neg|\+|\-)",
                text_lc,
            )
            if m6:
                val = _value_from_pnv_token(m6.group(1))
                if val:
                    _set_if_stronger(parsed, field, val)
                    break

        # Special-case NaCl tolerance with explicit percentages
        if field == "NaCl Tolerant (>=6%)":
            # We scan the whole text for positive/negative NaCl evidence,
            # then decide once per description. Negative has highest priority.
            has_positive = False
            has_negative = False

            # --- Negative phrasing (highest priority) ---
            # "does not grow in 7% NaCl", "doesn't grow at 10% NaCl"
            if re.search(
                r"does\s+(?:not|n't)\s+grow\s+(in|at)\s*\d+(?:\.\d+)?\s*%?\s*nacl",
                text_lc,
            ):
                has_negative = True

            # "no growth in 6.5% NaCl", "no growth at 8% NaCl"
            if re.search(
                r"no\s+growth\s+(in|at)\s*\d+(?:\.\d+)?\s*%?\s*nacl",
                text_lc,
            ):
                has_negative = True

            # "no growth in NaCl" (no explicit %)
            if re.search(
                r"no\s+growth\s+in\s+nacl",
                text_lc,
            ):
                has_negative = True

            # "unable to grow in 7% NaCl", "unable to grow in NaCl"
            if re.search(
                r"unable\s+to\s+grow\s+(in|at)\s*(\d+(?:\.\d+)?\s*%?\s*)?nacl",
                text_lc,
            ):
                has_negative = True

            # semantic negatives without explicit %
            if re.search(r"cannot\s+tolerate\s+nacl", text_lc):
                has_negative = True
            if re.search(r"not\s+nacl\s+tolerant", text_lc):
                has_negative = True
            if re.search(r"nacl\s+sensitive", text_lc):
                has_negative = True
            if re.search(r"fails\s+to\s+grow\s+(in|at)\s*(\d+(?:\.\d+)?\s*%?\s*)?nacl", text_lc):
                has_negative = True
            if re.search(r"intolerant\s+to\s+nacl", text_lc):
                has_negative = True
            if re.search(r"no\s+tolerance\s+to\s+nacl", text_lc):
                has_negative = True
            if re.search(r"nacl\s+intolerance", text_lc):
                has_negative = True
            if re.search(r"no\s+growth\s+at\s+high\s+nacl", text_lc):
                has_negative = True

            # --- Positive phrasing (requires growth/tolerance verb + % ≥ 6) ---
            # e.g. "grows in 6.5% NaCl", "growth occurs at 10% NaCl"
            for m in re.finditer(
                r"(grows|growth occurs|growth observed|able to grow|tolerates|tolerant)\s+"
                r"(?:in|at|up to|to)\s*(\d+(?:\.\d+)?)\s*%?\s*nacl",
                text_lc,
            ):
                try:
                    conc = float(m.group(2))
                    if conc >= 6.0:
                        has_positive = True
                except Exception:
                    pass

            # e.g. "NaCl tolerant up to 10%", "NaCl tolerant to 8%"
            for m in re.finditer(
                r"nacl\s+tolerant\s+(?:to|up to)?\s*(\d+(?:\.\d+)?)\s*%?",
                text_lc,
            ):
                try:
                    conc = float(m.group(1))
                    if conc >= 6.0:
                        has_positive = True
                except Exception:
                    pass

            # Decide final value:
            #   Negative > Positive > Unknown
            if has_negative:
                # Negative explicitly overrides any previous value
                parsed["NaCl Tolerant (>=6%)"] = "Negative"
            elif has_positive:
                _set_if_stronger(parsed, "NaCl Tolerant (>=6%)", "Positive")

    # Nitrate: "reduces nitrate" / "does not reduce nitrate"
    if re.search(r"reduces nitrate", text_lc):
        _set_if_stronger(parsed, "Nitrate Reduction", "Positive")
    if re.search(r"does (not|n't) reduce nitrate", text_lc):
        _set_if_stronger(parsed, "Nitrate Reduction", "Negative")

    # H2S: "produces H2S", "H2S production", "H2S production is positive"
    if re.search(r"(produces|production of)\s+h2s", text_lc):
        _set_if_stronger(parsed, "H2S", "Positive")
    if re.search(r"h2s production\s+is\s+(positive|pos|\+)", text_lc):
        _set_if_stronger(parsed, "H2S", "Positive")
    if re.search(r"h2s production\s+is\s+(negative|neg|\-)", text_lc):
        _set_if_stronger(parsed, "H2S", "Negative")
    if (
        re.search(r"does (not|n't) produce\s+h2s", text_lc)
        or re.search(r"no h2s production", text_lc)
        or re.search(r"non[- ]h2s producing", text_lc)
    ):
        _set_if_stronger(parsed, "H2S", "Negative")

    # --- DNase universal coverage ---
    # Positive forms
    if re.search(r"\bdnase(\s+test|\s+activity|\s+production)?\s*(positive|pos|\+)\b", text_lc):
        _set_if_stronger(parsed, "DNase", "Positive")

    if re.search(r"\b(positive|pos|\+)\s+dnase(\s+test|\s+activity|\s+production)?\b", text_lc):
        _set_if_stronger(parsed, "DNase", "Positive")

    # Negative forms
    if re.search(r"\bdnase(\s+test|\s+activity|\s+production)?\s*(negative|neg|\-)\b", text_lc):
        _set_if_stronger(parsed, "DNase", "Negative")

    if re.search(r"\b(negative|neg|\-)\s+dnase(\s+test|\s+activity|\s+production)?\b", text_lc):
        _set_if_stronger(parsed, "DNase", "Negative")

    # non-DNase-producing
    if re.search(r"\bnon[- ]?dnase[- ]?producing\b", text_lc):
        _set_if_stronger(parsed, "DNase", "Negative")

    # --- MR and VP grouped: "MR and VP negative" ---
    mr_vp_pattern = re.compile(
        r"\b("
        r"mr(?: test)?|methyl red|"
        r"vp(?: test)?|voges-proskauer"
        r")\s*(?:test)?\s*(?:and|&)\s*( "
        r"mr(?: test)?|methyl red|"
        r"vp(?: test)?|voges-proskauer"
        r")\s+"
        r"(positive|negative|variable|pos|neg|\+|\-)"
    )
    for m in mr_vp_pattern.finditer(text_lc):
        name1 = m.group(1)
        name2 = m.group(2)
        val = _value_from_pnv_token(m.group(3))
        if not val:
            continue

        def _assign_mr_vp(name: str) -> None:
            n = name.lower()
            if "mr" in n or "methyl red" in n:
                _set_if_stronger(parsed, "Methyl Red", val)
            if "vp" in n or "voges" in n:
                _set_if_stronger(parsed, "VP", val)

        _assign_mr_vp(name1)
        _assign_mr_vp(name2)

    # --- Decarboxylases global "all decarboxylases negative/positive" ---
    m_all_decarb = re.search(
        r"all\s+decarboxylases?\s+(?:are\s+)?(positive|negative|variable|pos|neg|\+|\-)",
        text_lc,
    )
    if m_all_decarb:
        val = _value_from_pnv_token(m_all_decarb.group(1))
        if val:
            for f in ("Lysine Decarboxylase", "Ornitihine Decarboxylase", "Arginine dihydrolase"):
                _set_if_stronger(parsed, f, val)

    # --- Generic grouped list logic for tests & sugars ---
    #
    # Handles things like:
    #   "gelatin and esculin hydrolysis negative"
    #   "lysine, ornithine and arginine negative"
    #   "indole, urease and citrate positive"
    #   "raffinose and inositol negative"
    #
    grouped_tests_pattern = re.compile(
        r"([a-z0-9 ,/&\-]+?)\s+"
        r"(?:hydrolysis|decarboxylases?|dihydrolases?|tests?|reactions?)?"
        r"\s*(?:are\s+)?(positive|negative|variable|pos|neg|\+|\-)"
    )

    for m in grouped_tests_pattern.finditer(text_lc):
        seg = m.group(1)
        val = _value_from_pnv_token(m.group(2))
        if not val:
            continue

        seg_lc = seg.lower()

        # Quick filter: does this segment contain any known test/sugar keyword?
        has_any = False

        for _, keywords in CORE_BOOL_FIELDS.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", seg_lc) for kw in keywords):
                has_any = True
                break

        if not has_any:
            for sugar_key in SUGAR_FIELDS.keys():
                if re.search(rf"\b{sugar_key}\b", seg_lc):
                    has_any = True
                    break

        if not has_any:
            continue  # ignore segments unrelated to tests/sugars

        # Apply to all matching core boolean tests
        for field, keywords in CORE_BOOL_FIELDS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", seg_lc):
                    _set_if_stronger(parsed, field, val)
                    break

        # Apply to all matching sugars
        for sugar_key, field in SUGAR_FIELDS.items():
            if re.search(rf"\b{sugar_key}\b", seg_lc):
                _set_if_stronger(parsed, field, val)


# ------------------------------------------------------------
# Motility / Capsule / Spores
# ------------------------------------------------------------

def _parse_motility_capsule_spores(text_lc: str, parsed: Dict[str, str]) -> None:
    # Motility
    if (
        re.search(r"\bmotile\b", text_lc)
        and not re.search(r"\bnon[- ]?motile\b", text_lc)
        and "nonmotile" not in text_lc
        and "immotile" not in text_lc
    ):
        _set_if_stronger(parsed, "Motility", "Positive")

    if (
        "non-motile" in text_lc
        or "non motile" in text_lc
        or "nonmotile" in text_lc
        or "immotile" in text_lc
    ):
        _set_if_stronger(parsed, "Motility", "Negative")

    # Specific motility phrases: tumbling, swarming, corkscrew, gliding, spreading
    if (
        "tumbling motility" in text_lc
        or "swarming motility" in text_lc
        or "corkscrew motility" in text_lc
        or re.search(r"\b(gliding|spreading)\s+motility\b", text_lc)
        or ("swarming" in text_lc and "non-swarming" not in text_lc)
    ):
        _set_if_stronger(parsed, "Motility", "Positive")

    # Capsule (including "capsule positive/negative", present/absent)
    if (
        "capsulated" in text_lc
        or "encapsulated" in text_lc
        or "capsule present" in text_lc
        or re.search(r"capsule\s+is\s+present", text_lc)
        or re.search(r"capsule[ \-]?(positive|pos|\+)", text_lc)
    ):
        _set_if_stronger(parsed, "Capsule", "Positive")

    if (
        "non-capsulated" in text_lc
        or "no capsule" in text_lc
        or "capsule absent" in text_lc
        or re.search(r"capsule\s+is\s+absent", text_lc)
        or re.search(r"capsule[ \-]?(negative|neg|\-)", text_lc)
    ):
        _set_if_stronger(parsed, "Capsule", "Negative")

    # Spore formation
    # NEGATIVE FIRST with strict boundaries, then early-return
    if (
        re.search(r"\bnon[-\s]?spore[-\s]?forming\b", text_lc)
        or "no spores" in text_lc
    ):
        _set_if_stronger(parsed, "Spore Formation", "Negative")
        return  # prevent any positive overwrite

    # POSITIVE (must not match the negative form)
    if (
        re.search(r"\bspore[-\s]?forming\b", text_lc)
        or "forms spores" in text_lc
    ):
        _set_if_stronger(parsed, "Spore Formation", "Positive")


# ------------------------------------------------------------
# Oxygen requirement
# ------------------------------------------------------------

def _parse_oxygen(text_lc: str, parsed: Dict[str, str]) -> None:
    """
    Robust oxygen parsing:
      - Handle facultative first
      - Avoid "aerobic" accidentally matching inside "anaerobic"
      - Include "aerobically" / "anaerobically"
    """
    # Facultative first
    if re.search(r"facultative(ly)? anaerob", text_lc):
        _set_if_stronger(parsed, "Oxygen Requirement", "Facultative Anaerobe")

    # Strict anaerobic (before aerobic)
    if (
        re.search(r"\bobligate anaerob", text_lc)
        or (re.search(r"\banaerobic\b", text_lc) and "facultative" not in text_lc)
        or re.search(r"\banaerobically\b", text_lc)
    ):
        _set_if_stronger(parsed, "Oxygen Requirement", "Anaerobic")

    # Now handle purely aerobic, avoiding "anaerobic"
    if (
        re.search(r"\bobligate aerobe\b", text_lc)
        or (re.search(r"\baerobic\b", text_lc) and "anaerobic" not in text_lc)
        or (
            re.search(r"\baerobically\b", text_lc)
            and "anaerobically" not in text_lc
        )
    ):
        _set_if_stronger(parsed, "Oxygen Requirement", "Aerobic")

    if "microaerophilic" in text_lc or "microaerophile" in text_lc:
        _set_if_stronger(parsed, "Oxygen Requirement", "Microaerophilic")

    if "capnophilic" in text_lc or "co2" in text_lc:
        _set_if_stronger(parsed, "Oxygen Requirement", "Capnophilic")


# ------------------------------------------------------------
# Growth temperature
# ------------------------------------------------------------

def _parse_growth_temperature(text_lc: str, parsed: Dict[str, str]) -> None:
    """
    Look for explicit temperatures like "grows at 37 °C" or ranges like "4–45 °C".
    We ALWAYS store as "low//high":
      - true ranges: "4-45 °C" → "4//45"
      - "grows between 30 and 37 °C" → "30//37"
      - "grows at 30–37 °C" → "30//37"
      - two temps in text: min//max (Option A)
      - single temps: "37 °C" → "37//37"
    """
    # 0) Explicit "between X and Y" ranges
    between_pattern = re.compile(
        r"between\s+(\d+)\s*(?:c|°c|degrees c|degrees celsius)?"
        r"\s*(?:and|to|-)\s*(\d+)\s*(?:c|°c|degrees c|degrees celsius)?"
    )
    m_between = between_pattern.search(text_lc)
    if m_between:
        low = m_between.group(1)
        high = m_between.group(2)
        _set_if_stronger(parsed, "Growth Temperature", f"{low}//{high}")
        return

    # 1) Explicit ranges like "4-45 °C" or "10–40 °C"
    range_pattern = re.compile(
        r"(\d+)\s*[-–/]\s*(\d+)\s*(?:c|°c|degrees c|degrees celsius)"
    )
    m_range = range_pattern.search(text_lc)
    if m_range:
        low = m_range.group(1)
        high = m_range.group(2)
        _set_if_stronger(parsed, "Growth Temperature", f"{low}//{high}")
        return

    # 2) Any two explicit temps → min//max
    temps = re.findall(r"(\d+)\s*(?:c|°c|degrees c|degrees celsius)", text_lc)
    if len(temps) >= 2:
        nums = [int(t) for t in temps]
        low = min(nums)
        high = max(nums)
        _set_if_stronger(parsed, "Growth Temperature", f"{low}//{high}")
        return

    # 3) Single temps like "grows at 37 c"
    single_pattern = re.compile(
        r"(grows|growth|optimum|optimal)\s+(?:at\s+)?(\d+)\s*"
        r"(?:c|°c|degrees c|degrees celsius)"
    )
    m_single = single_pattern.search(text_lc)
    if m_single:
        temp = m_single.group(2)
        _set_if_stronger(parsed, "Growth Temperature", f"{temp}//{temp}")
        return

    # 4) Simplified: "grows at 37" (no explicit °C)
    m_simple_num = re.search(r"grows at (\d+)\b", text_lc)
    if m_simple_num:
        temp = m_simple_num.group(1)
        _set_if_stronger(parsed, "Growth Temperature", f"{temp}//{temp}")
        return

    # 5) Fallback: plain "37c" somewhere in the text
    m_plain = re.search(
        r"\b(\d+)\s*(?:c|°c|degrees c|degrees celsius)\b",
        text_lc,
    )
    if m_plain:
        temp = m_plain.group(1)
        _set_if_stronger(parsed, "Growth Temperature", f"{temp}//{temp}")


# ------------------------------------------------------------
# Media grown on (coarse mapping)
# ------------------------------------------------------------

MEDIA_KEYWORDS = {
    "Blood Agar": [
        "blood agar",
        "blood-agar",
    ],
    "MacConkey Agar": [
        "macconkey agar",
        "mac conkey agar",
        "macconkey",
    ],
    "Chocolate Agar": [
        "chocolate agar",
        "chocolate-agar",
    ],
    "Nutrient Agar": [
        "nutrient agar",
        "nutrient-agar",
        "nut agar",
    ],
    "XLD Agar": [
        "xld agar",
        "xld",
    ],
    "TCBS Agar": [
        "tcbs agar",
        "tcbs",
    ],
    "ALOA": [
        "aloa agar",
        "aloa",
    ],
    "BCYE Agar": [
        "bcye agar",
        "bcye",
        "Buffered Charcoal Yeast Extract Agar",
        "buffered charcoal yeast extract agar"
    ],
    "MRS Agar": [
        "mrs agar",
    ],
    "Mannitol Salt Agar": [
        "msa agar",
        "ms agar",
    ],
    "Cycloserine Cefoxitin Fructose Agar": [
        "ccfa agar",
        "cycloserine cefoxitin fructose agar",
        "ccf agar",
    ],
    "Thayer Martin Agar": [
        "thayer martin agar",
        "tma agar",
        "tma",
    ],
    "Bordet-Gengou Agar": [
        "bordet gengou agar",
    ],
    "Cetrimide Agar": [
        "cetrimide agar",
    ],
    "Anaerobic Agar": [
        "anaerobic agar",
    ],
    "Anaerobic Blood Agar": [
        "anaerobic blood agar",
    ],
    "Hektoen Enteric Agar": [
        "hektoen enteric agar",
        "HK Agar",
        "hk",
    ],
    "Tryptic Soy Agar": [
        "tryptic soy agar",
        "t-soy agar",
        "tsoy",
    ],
    "Brucella Agar": [
        "brucella agar",
    ],
    "Charcoal Agar": [
        "charcoal agar",
    ],
    "Yeast Extract Mannitol Agar": [
        "yeast extract mannitol agar",
    ],
    "Sabouraud Agar": [
        "sabouraud agar",
        "sabouraud dextrose agar",
    ],
    "BHI": [
        "bhi",
        "brain heart infusion agar",
        "brain heart infusion",
    ],
    "Columbia Blood Agar": [
        "columbia blood agar",
        "columbia agar",
        "columbia",
    ],
    "Lowenstein-Jensen Agar": [
        "lowenstein-jensen agar",
        "lowenstein jensen agar",
    ],
    "BSK Medium": [
        "bsk medium",
        "bsk",
        "bsk-ii medium",
        "bsk-h medium",
    ],
    "Ashby Agar": [
        "ashby agar",
        "ashby medium",
    ]
}


def _parse_media(text_lc: str, parsed: Dict[str, str]) -> None:
    found_media: List[str] = []
    for media_name, patterns in MEDIA_KEYWORDS.items():
        for p in patterns:
            if p in text_lc and media_name not in found_media:
                found_media.append(media_name)

    if found_media:
        _set_if_stronger(parsed, "Media Grown On", "; ".join(found_media))


# ------------------------------------------------------------
# Sugar fermentation parsing
# ------------------------------------------------------------

def _parse_sugars(text_lc: str, parsed: Dict[str, str]) -> None:
    """
    Handles patterns like:
      - "glucose positive, mannitol negative"
      - "ferments glucose, mannitol and sucrose but not lactose"
      - "does not ferment lactose or sucrose"
      - "non-lactose fermenter"
      - "<sugar> fermenter" (positive unless "non-<sugar> fermenter")
      - "<sugar> is positive/negative"
      - "<sugar> fermentation is positive/negative"
      - global non-fermenter phrases
      - "asaccharolytic" → all sugars Negative (Unknown-only)
      - "all other sugars negative" → remaining sugars Negative
    """

    # 0) Simple "<sugar> positive/negative" and "<sugar> is positive"
    for sugar_key, field in SUGAR_FIELDS.items():
        # "glucose positive"
        m_simple = re.search(
            rf"{sugar_key}\s+(positive|negative|variable|pos|neg|\+|\-)",
            text_lc,
        )
        if m_simple:
            val = _value_from_pnv_context(m_simple.group(1))
            if val:
                _set_if_stronger(parsed, field, val)

        # "<sugar> is positive"
        m_is = re.search(
            rf"{sugar_key}\s+is\s+(positive|negative|variable|pos|neg|\+|\-)",
            text_lc,
        )
        if m_is:
            val = _value_from_pnv_token(m_is.group(1))
            if val:
                _set_if_stronger(parsed, field, val)

    # 0b) "<sugar> fermenter" vs "non-<sugar> fermenter"
    for sugar_key, field in SUGAR_FIELDS.items():
        # positive: "lactose fermenter"
        if re.search(rf"\b{sugar_key}\s+fermenter\b", text_lc) and not re.search(
            rf"\bnon[- ]{sugar_key}\s+fermenter\b", text_lc
        ):
            _set_if_stronger(parsed, field, "Positive")

        # negative: "non-lactose fermenter"
        if re.search(rf"\bnon[- ]{sugar_key}\s+fermenter\b", text_lc):
            _set_if_stronger(parsed, field, "Negative")

    # 1) "ferments X, Y and Z but not A, B"
    ferments_pattern = re.compile(r"ferments\s+([a-z0-9 ,;/&\-]+)")
    for m in ferments_pattern.finditer(text_lc):
        seg = m.group(1)
        # Split positive vs negative part on "but not"
        neg_split = re.split(r"\bbut not\b", seg, maxsplit=1)
        pos_part = neg_split[0]
        neg_part = neg_split[1] if len(neg_split) > 1 else ""

        # Positive sugars from pos_part
        for sugar_key, field in SUGAR_FIELDS.items():
            if re.search(rf"\b{sugar_key}\b", pos_part):
                _set_if_stronger(parsed, field, "Positive")

        # Negative sugars from neg_part
        for sugar_key, field in SUGAR_FIELDS.items():
            if re.search(rf"\b{sugar_key}\b", neg_part):
                _set_if_stronger(parsed, field, "Negative")

    # 2) Grouped "does not ferment X, Y and Z" (stop at but/punctuation)
    #    Prevents glucose being accidentally marked negative in:
    #      "does not ferment lactose or sucrose, but glucose fermentation is positive"
    grouped_neg_pattern = re.compile(
        r"does\s+(?:not|n't)\s+ferment\s+([a-z0-9 ,;/&\-]+?)(?:\s+but\b|\.|;|,|$)"
    )
    for m in grouped_neg_pattern.finditer(text_lc):
        seg = m.group(1)
        for sugar_key, field in SUGAR_FIELDS.items():
            if re.search(rf"\b{sugar_key}\b", seg):
                _set_if_stronger(parsed, field, "Negative")

    # 3) Single "does not ferment X"
    for sugar_key, field in SUGAR_FIELDS.items():
        if re.search(
            rf"does\s+(?:not|n't)\s+ferment\s+{sugar_key}\b", text_lc
        ):
            _set_if_stronger(parsed, field, "Negative")

    # 4) "non-lactose fermenter" and similar
    for sugar_key, field in SUGAR_FIELDS.items():
        if re.search(
            rf"non[- ]{sugar_key}\s+ferment(ing|er)?", text_lc
        ):
            _set_if_stronger(parsed, field, "Negative")

    # 5) "<sugar> fermentation positive/negative" + "is positive"
    for sugar_key, field in SUGAR_FIELDS.items():
        # "glucose fermentation positive"
        m1 = re.search(
            rf"{sugar_key}\s+fermentation[ \-]?"
            r"(positive|negative|variable|pos|neg|\+|\-)",
            text_lc,
        )
        if m1:
            val = _value_from_pnv_context(m1.group(1))
            if val:
                _set_if_stronger(parsed, field, val)
                continue

        # "positive for glucose fermentation"
        m2 = re.search(
            rf"(positive|negative|variable|pos|neg|\+|\-)\s+"
            rf"(for\s+)?{sugar_key}\s+fermentation",
            text_lc,
        )
        if m2:
            val = _value_from_pnv_context(m2.group(1))
            if val:
                _set_if_stronger(parsed, field, val)
                continue

        # "<sugar> fermentation is positive/negative"
        m3 = re.search(
            rf"{sugar_key}\s+fermentation\s+is\s+"
            r"(positive|negative|variable|pos|neg|\+|\-)",
            text_lc,
        )
        if m3:
            val = _value_from_pnv_token(m3.group(1))
            if val:
                _set_if_stronger(parsed, field, val)
                continue

    # 6) Global non-fermenter phrases
    #     e.g. "non-fermenter", "does not ferment sugars"
    #     → set all sugars Negative *unless* already set by a more specific rule.
    if (
        re.search(
            r"does\s+(?:not|n't)\s+ferment\s+(carbohydrates|sugars)", text_lc
        )
        or re.search(r"\bnon[- ]ferment(er|ing|ative)\b", text_lc)
    ):
        for field in SUGAR_FIELDS.values():
            if field not in parsed or parsed[field] == UNKNOWN:
                _set_if_stronger(parsed, field, "Negative")

    # 7) Asaccharolytic → all sugars Negative (Unknown-only)
    if (
        "asaccharolytic" in text_lc
        or "non-saccharolytic" in text_lc
        or "non saccharolytic" in text_lc
    ):
        for field in SUGAR_FIELDS.values():
            if field not in parsed or parsed[field] == UNKNOWN:
                _set_if_stronger(parsed, field, "Negative")

    # 8) "all other sugars negative/positive"
    m_other = re.search(
        r"all\s+other\s+sugars\s+(?:are\s+)?(positive|negative|variable|pos|neg|\+|\-)",
        text_lc,
    )
    if m_other:
        val = _value_from_pnv_token(m_other.group(1))
        if val:
            for field in SUGAR_FIELDS.values():
                if field not in parsed or parsed[field] == UNKNOWN:
                    _set_if_stronger(parsed, field, val)


# ------------------------------------------------------------
# Colony morphology (coarse, optional)
# ------------------------------------------------------------

def _normalise_colony_desc(desc: str) -> str:
    """
    Take a raw colony descriptor and normalise into:
      "Smooth; Yellow; Opaque" etc.

    Tweaks:
      - Remove "-pigmented" → "yellow-pigmented" → "yellow"
      - Treat "and" like a separator for parts
    """
    # Remove "-pigmented" so "yellow-pigmented" → "yellow"
    tmp = desc.replace("-pigmented", "")

    # Normalise "and" to a comma so it acts like a separator
    tmp = tmp.replace(" and ", ", ")

    parts = [s.strip() for s in re.split(r"[;,]", tmp) if s.strip()]
    pretty = "; ".join(p.capitalize() for p in parts)
    return pretty


def _parse_colony(text_lc: str, parsed: Dict[str, str]) -> None:
    """
    Very coarse mapping for colony morphology. We try:
      - "colonies are yellow, mucoid"
      - "colonies dry, white and irregular on nutrient agar"
      - "forming smooth, yellow-pigmented, opaque colonies"
      - "grey colonies", "large grey colonies" (no verb)
    """

    # Pattern 1: "colonies are ..."
    m = re.search(r"colon(y|ies)\s+(are|is)\s+([a-z0-9 ,;\-]+)", text_lc)
    if m:
        desc = m.group(3).strip()
        if desc:
            pretty = _normalise_colony_desc(desc)
            if pretty:
                _set_if_stronger(parsed, "Colony Morphology", pretty)
                return

    # Pattern 2: "colonies dry, white and irregular on nutrient agar"
    m2 = re.search(
        r"colonies\s+([a-z0-9 ,;\-]+?)(?:\s+on\b|\.|,)",
        text_lc,
    )
    if m2:
        desc = m2.group(1).strip()
        if desc:
            pretty = _normalise_colony_desc(desc)
            if pretty:
                _set_if_stronger(parsed, "Colony Morphology", pretty)
                return

    # Pattern 3: "forming green colonies", "forms mucoid colonies",
    #            "forming smooth, yellow-pigmented, opaque colonies"
    m3 = re.search(
        r"(forming|forms|produces)\s+([a-z0-9 ,;\-]+?)\s+colonies",
        text_lc,
    )
    if m3:
        desc = m3.group(2).strip()
        if desc:
            pretty = _normalise_colony_desc(desc)
            if pretty:
                _set_if_stronger(parsed, "Colony Morphology", pretty)
                return

    # Pattern 4: plain descriptor before "colonies" (e.g. "grey colonies",
    # "large grey colonies") when none of the above match.
    m4 = re.search(
        r"\b([a-z0-9 ,;\-]+?)\s+colonies\b",
        text_lc,
    )
    if m4:
        desc = m4.group(1).strip()
        if desc:
            pretty = _normalise_colony_desc(desc)
            if pretty:
                _set_if_stronger(parsed, "Colony Morphology", pretty)
                return

def _apply_patches(original_text: str, text_lc: str, parsed: Dict[str, str]) -> Dict[str, str]:
    # ----------------------------------------------
    # helper for P/N/V
    # ----------------------------------------------
    def _pnv(x: str) -> Optional[str]:
        x = x.strip().lower()
        if x in {"positive", "pos", "+", "strongly positive", "weakly positive"}:
            return "Positive"
        if x in {"negative", "neg", "-", "no"}:
            return "Negative"
        if x in {"variable", "var", "mixed"}:
            return "Variable"
        return None

    # ============================================================
    #  NEW LOGIC: Haemolysis Type detection (alpha/beta/none)
    # ============================================================

    # alpha
    m_alpha = re.search(r"(alpha|α)[-\s]*haemolysis", text_lc) or \
              re.search(r"haemolysis type[: ]*(alpha|α)", text_lc)
    if m_alpha:
        if parsed.get("Haemolysis", UNKNOWN) == UNKNOWN:
            parsed["Haemolysis"] = "Positive"
        if parsed.get("Haemolysis Type", UNKNOWN) == UNKNOWN:
            parsed["Haemolysis Type"] = "Alpha"

    # beta
    m_beta = re.search(r"(beta|β)[-\s]*haemolysis", text_lc) or \
             re.search(r"haemolysis type[: ]*(beta|β)", text_lc)
    if m_beta:
        if parsed.get("Haemolysis", UNKNOWN) == UNKNOWN:
            parsed["Haemolysis"] = "Positive"
        if parsed.get("Haemolysis Type", UNKNOWN) == UNKNOWN:
            parsed["Haemolysis Type"] = "Beta"

    # gamma / none
    m_gamma = re.search(r"(gamma|γ)[-\s]*haemolysis", text_lc)
    m_none = re.search(r"(no haemolysis|non[- ]haemolytic|no hemolysis|non[- ]hemolytic)", text_lc)
    if m_gamma or m_none:
        if parsed.get("Haemolysis", UNKNOWN) == UNKNOWN:
            parsed["Haemolysis"] = "Negative"
        if parsed.get("Haemolysis Type", UNKNOWN) == UNKNOWN:
            parsed["Haemolysis Type"] = "None"

    # ============================================================
    # ORIGINAL PATCH v1 LOGIC (fully preserved)
    # ============================================================

    # 1. Haemolysis: generic ± without type
    m_h = re.search(r"haemolysis\s+(positive|negative|variable|pos|neg|\+|\-)", text_lc)
    if m_h and "Haemolysis" not in parsed:
        val = _pnv(m_h.group(1))
        if val:
            parsed["Haemolysis"] = val
            if parsed.get("Haemolysis Type", UNKNOWN) == UNKNOWN and val == "Positive":
                parsed["Haemolysis Type"] = "Unknown"

    # 2. Motility: generic ±
    m_mot = re.search(r"motility\s+(positive|negative|variable|pos|neg|\+|\-)", text_lc)
    if m_mot and "Motility" not in parsed:
        val = _pnv(m_mot.group(1))
        if val:
            parsed["Motility"] = val

    # 3. Spore formation ±
    m_sp = re.search(r"spore formation\s+(positive|negative|variable|pos|neg|\+|\-)", text_lc)
    if m_sp and parsed.get("Spore Formation", UNKNOWN) == UNKNOWN:
        val = _pnv(m_sp.group(1))
        if val:
            parsed["Spore Formation"] = val

    # ============================================================
    #  FIXED NaCl tolerant logic (patch upgrade)
    # ============================================================
    if parsed.get("NaCl Tolerant (>=6%)", UNKNOWN) == UNKNOWN:

        # direct p/n/v
        m_nacl = re.search(
            r"(?:nacl\s*(?:tolerant|tolerance)?|growth\s+in\s+6\%[\s]*nacl)"
            r"\s*(positive|negative|variable|pos|neg|\+|\-)",
            text_lc
        )
        if m_nacl:
            val = _pnv(m_nacl.group(1))
            if val:
                parsed["NaCl Tolerant (>=6%)"] = val

        # "no growth in 6% nacl"
        if parsed.get("NaCl Tolerant (>=6%)", UNKNOWN) == UNKNOWN:
            if re.search(r"no\s+growth\s+in\s+(?:>=)?\s*6\%?\s*nacl", text_lc):
                parsed["NaCl Tolerant (>=6%)"] = "Negative"

        # "grows in 6% nacl"
        if parsed.get("NaCl Tolerant (>=6%)", UNKNOWN) == UNKNOWN:
            if re.search(r"grows?\s+in\s+(?:>=)?\s*6\%?\s*nacl", text_lc):
                parsed["NaCl Tolerant (>=6%)"] = "Positive"

    # ============================================================
    #  Growth Temperature patterns (20/40, 20//40, 20 / 40)
    # ============================================================
    m_temp = re.search(r"\b(\d{1,3})\s*[/]{1,2}\s*(\d{1,3})\b", text_lc)
    if m_temp and parsed.get("Growth Temperature", UNKNOWN) == UNKNOWN:
        parsed["Growth Temperature"] = f"{m_temp.group(1)}//{m_temp.group(2)}"

    # ============================================================
    #  Colony Morphology STRICT LIST extraction
    # ============================================================
    COLONY_TRIGGERS = [
        "colony morphology",
        "colonies are",
        "colonies appear",
        "colonies look",
        "colony appearance",
        "colony characteristics",
    ]
    if any(t in text_lc for t in COLONY_TRIGGERS):
        m_col = re.search(
            r"(?:colony morphology|colonies are|colonies appear|colonies look|colony appearance|colony characteristics)"
            r"[: ]+([a-z0-9 ,;/\-]+)",
            text_lc
        )
        if m_col:
            segment = m_col.group(1)
            parts = [x.strip() for x in re.split(r"[;,/]", segment) if x.strip()]

            clean_desc = [p.capitalize() for p in parts if len(p) > 1]

            if clean_desc:
                existing = parsed.get("Colony Morphology", "")
                existing_list = [x.strip() for x in existing.split(";")] if existing else []

                merged = []
                for x in existing_list:
                    if x not in merged:
                        merged.append(x)
                for x in clean_desc:
                    if x not in merged:
                        merged.append(x)

                parsed["Colony Morphology"] = "; ".join(merged)

    # ============================================================
    #  ORIGINAL MULTI-MEDIA PATCH (unchanged)
    # ============================================================
    if "media grown on" in text_lc or "grown on" in text_lc:
        mm = re.search(r"(?:media\s+grown\s+on|grown\s+on)[: ]+([a-z0-9 ,;/\-]+)", text_lc)
        if mm:
            segment = mm.group(1)
            raw_items = re.split(r"[;,]", segment)
            raw_items = [x.strip() for x in raw_items if x.strip()]

            detected_media = []
            for item in raw_items:
                for media_name, patterns in MEDIA_KEYWORDS.items():
                    for p in patterns:
                        if p in item and media_name not in detected_media:
                            detected_media.append(media_name)

            if detected_media:
                existing = parsed.get("Media Grown On", "")
                existing_list = [x.strip() for x in existing.split(";")] if existing else []

                merged = []
                for m in existing_list:
                    if m not in merged:
                        merged.append(m)
                for m in detected_media:
                    if m not in merged:
                        merged.append(m)

                parsed["Media Grown On"] = "; ".join(merged)

    return parsed

# ------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------

def parse_text_rules(text: str) -> Dict[str, Any]:
    """
    Main entry point for the rule-based core parser.
    """
    original = text or ""
    text_clean = _clean_text(original)
    text_lc = text_clean.lower()

    parsed: Dict[str, str] = {}

    try:
        _parse_gram_and_shape(text_lc, parsed)
        _parse_haemolysis(text_lc, parsed)
        _parse_core_bool_tests(text_lc, parsed)
        _parse_motility_capsule_spores(text_lc, parsed)
        _parse_oxygen(text_lc, parsed)
        _parse_growth_temperature(text_lc, parsed)
        _parse_media(text_lc, parsed)
        _parse_sugars(text_lc, parsed)
        _parse_colony(text_lc, parsed)
        parsed = _apply_patches(original, text_lc, parsed)
        
        return {
            "parsed_fields": parsed,
            "source": "rule_parser",
            "raw": original,
        }

    except Exception as e:
        # Fail-safe: never crash the app, just report an error
        return {
            "parsed_fields": parsed,
            "source": "rule_parser",
            "raw": original,
            "error": f"{type(e).__name__}: {e}",
        }
