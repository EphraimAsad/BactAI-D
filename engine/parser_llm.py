# engine/parser_llm.py
# ------------------------------------------------------------
# Local LLM parser for BactAI-D (Flan-T5, CPU-friendly)
# Third parser head: repair & recovery
#
# Drop-in patched version:
# - Few-shot examples increased (configurable via env)
# - Field alias mapping (prevents silent field drops)
# - Non-greedy JSON extraction (prevents regex over-capture)
# - Improved P/N/V normalization (Flan phrasing coverage)
# - Prompt refined for "extract/clarify" (reduces Unknown collapse)
# - Debug prints (toggle via env var)
# - Sugar logic scaffold preserved
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

DEFAULT_MODEL = os.getenv(
    "BACTAI_LLM_PARSER_MODEL",
    "google/flan-t5-base",
)

MAX_FEWSHOT_EXAMPLES = int(os.getenv("BACTAI_LLM_FEWSHOT", "25"))
MAX_NEW_TOKENS = int(os.getenv("BACTAI_LLM_MAX_NEW_TOKENS", "128"))

DEBUG_LLM = os.getenv("BACTAI_LLM_DEBUG", "1").strip().lower() in {
    "1", "true", "yes", "y", "on"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSeq2SeqLM] = None
_GOLD_EXAMPLES: Optional[List[Dict[str, Any]]] = None


# ------------------------------------------------------------
# Allowed fields
# ------------------------------------------------------------

ALL_FIELDS: List[str] = [
    "Gram Stain",
    "Shape",
    "Motility",
    "Capsule",
    "Spore Formation",
    "Haemolysis",
    "Haemolysis Type",
    "Media Grown On",
    "Colony Morphology",
    "Oxygen Requirement",
    "Growth Temperature",
    "Catalase",
    "Oxidase",
    "Indole",
    "Urease",
    "Citrate",
    "Methyl Red",
    "VP",
    "H2S",
    "DNase",
    "ONPG",
    "Coagulase",
    "Gelatin Hydrolysis",
    "Esculin Hydrolysis",
    "Nitrate Reduction",
    "NaCl Tolerant (>=6%)",
    "Lipase Test",
    "Lysine Decarboxylase",
    "Ornithine Decarboxylase",
    "Ornitihine Decarboxylase",
    "Arginine dihydrolase",
    "Glucose Fermentation",
    "Lactose Fermentation",
    "Sucrose Fermentation",
    "Maltose Fermentation",
    "Mannitol Fermentation",
    "Sorbitol Fermentation",
    "Xylose Fermentation",
    "Rhamnose Fermentation",
    "Arabinose Fermentation",
    "Raffinose Fermentation",
    "Trehalose Fermentation",
    "Inositol Fermentation",
    "Gas Production",
    "TSI Pattern",
    "Colony Pattern",
    "Pigment",
    "Motility Type",
    "Odor",
]

SUGAR_FIELDS = [
    "Glucose Fermentation",
    "Lactose Fermentation",
    "Sucrose Fermentation",
    "Maltose Fermentation",
    "Mannitol Fermentation",
    "Sorbitol Fermentation",
    "Xylose Fermentation",
    "Rhamnose Fermentation",
    "Arabinose Fermentation",
    "Raffinose Fermentation",
    "Trehalose Fermentation",
    "Inositol Fermentation",
]

PNV_FIELDS = {
    f for f in ALL_FIELDS
    if f not in {
        "Media Grown On",
        "Colony Morphology",
        "Growth Temperature",
        "Gram Stain",
        "Shape",
        "Oxygen Requirement",
        "Haemolysis Type",
    }
}


# ------------------------------------------------------------
# Field alias mapping (CRITICAL)
# ------------------------------------------------------------

FIELD_ALIASES: Dict[str, str] = {
    "Gram": "Gram Stain",
    "Gram stain": "Gram Stain",
    "Gram Stain Result": "Gram Stain",

    "NaCl tolerance": "NaCl Tolerant (>=6%)",
    "NaCl Tolerant": "NaCl Tolerant (>=6%)",
    "Salt tolerance": "NaCl Tolerant (>=6%)",
    "Salt tolerant": "NaCl Tolerant (>=6%)",
    "6.5% NaCl": "NaCl Tolerant (>=6%)",
    "6% NaCl": "NaCl Tolerant (>=6%)",

    "Growth temp": "Growth Temperature",
    "Growth temperature": "Growth Temperature",
    "Temperature growth": "Growth Temperature",

    "Catalase test": "Catalase",
    "Oxidase test": "Oxidase",
    "Indole test": "Indole",
    "Urease test": "Urease",
    "Citrate test": "Citrate",

    "Glucose fermentation": "Glucose Fermentation",
    "Lactose fermentation": "Lactose Fermentation",
    "Sucrose fermentation": "Sucrose Fermentation",
    "Maltose fermentation": "Maltose Fermentation",
    "Mannitol fermentation": "Mannitol Fermentation",
    "Sorbitol fermentation": "Sorbitol Fermentation",
    "Xylose fermentation": "Xylose Fermentation",
    "Rhamnose fermentation": "Rhamnose Fermentation",
    "Arabinose fermentation": "Arabinose Fermentation",
    "Raffinose fermentation": "Raffinose Fermentation",
    "Trehalose fermentation": "Trehalose Fermentation",
    "Inositol fermentation": "Inositol Fermentation",
}


# ------------------------------------------------------------
# Normalisation helpers
# ------------------------------------------------------------

def _norm_str(s: Any) -> str:
    return str(s).strip() if s is not None else ""


def _normalise_pnv_value(raw: Any) -> str:
    s = _norm_str(raw).lower()
    if not s:
        return "Unknown"

    if any(x in s for x in {"positive", "pos", "+", "yes", "present", "detected", "reactive"}):
        return "Positive"

    if any(x in s for x in {"negative", "neg", "-", "no", "none", "absent", "not detected", "no growth"}):
        return "Negative"

    if any(x in s for x in {"variable", "mixed", "inconsistent"}):
        return "Variable"

    return "Unknown"


def _normalise_gram(raw: Any) -> str:
    s = _norm_str(raw).lower()
    if "positive" in s:
        return "Positive"
    if "negative" in s:
        return "Negative"
    if "variable" in s:
        return "Variable"
    return "Unknown"


def _merge_ornithine_variants(fields: Dict[str, str]) -> Dict[str, str]:
    v = fields.get("Ornithine Decarboxylase") or fields.get("Ornitihine Decarboxylase")
    if v and v != "Unknown":
        fields["Ornithine Decarboxylase"] = v
        fields["Ornitihine Decarboxylase"] = v
    return fields


# ------------------------------------------------------------
# Sugar logic
# ------------------------------------------------------------

_NON_FERMENTER_PATTERNS = re.compile(
    r"\b("
    r"non[-\s]?fermenter|"
    r"non[-\s]?fermentative|"
    r"asaccharolytic|"
    r"does not ferment (sugars|carbohydrates)|"
    r"no carbohydrate fermentation"
    r")\b",
    re.IGNORECASE,
)


def _apply_global_sugar_logic(fields: Dict[str, str], original_text: str) -> Dict[str, str]:
    if not _NON_FERMENTER_PATTERNS.search(original_text):
        return fields

    for sugar in SUGAR_FIELDS:
        if fields.get(sugar) in {"Positive", "Variable"}:
            continue
        fields[sugar] = "Negative"

    return fields


# ------------------------------------------------------------
# Gold examples
# ------------------------------------------------------------

def _get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_gold_examples() -> List[Dict[str, Any]]:
    global _GOLD_EXAMPLES
    if _GOLD_EXAMPLES is not None:
        return _GOLD_EXAMPLES

    path = os.path.join(_get_project_root(), "data", "llm_gold_examples.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _GOLD_EXAMPLES = data if isinstance(data, list) else []
    except Exception:
        _GOLD_EXAMPLES = []

    return _GOLD_EXAMPLES


# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

PROMPT_HEADER = """
You are a microbiology expert assisting an automated phenotype parser.

Your task is to EXTRACT OR CLARIFY phenotypic and biochemical test results
from the input text.

Rules:
- Return ONLY valid JSON
- Do NOT invent results
- If a result is unclear or not stated, use "Unknown"
- Prefer explicit statements over assumptions

Output format:
{
  "parsed_fields": {
    "Field Name": "Value",
    ...
  }
}
"""

PROMPT_FOOTER = """
Now process the following phenotype description.

Input:
\"\"\"<<PHENOTYPE>>\"\"\"

Return ONLY the JSON object.
"""


def _build_prompt(text: str) -> str:
    examples = _load_gold_examples()
    n = min(MAX_FEWSHOT_EXAMPLES, len(examples))
    sampled = random.sample(examples, n) if n > 0 else []

    blocks: List[str] = [PROMPT_HEADER]

    for ex in sampled:
        inp = _norm_str(ex.get("input", ""))
        exp = ex.get("expected", {})
        if not isinstance(exp, dict):
            exp = {}

        blocks.append(
            f'Input:\n"""{inp}"""\nOutput:\n'
            f'{json.dumps({"parsed_fields": exp}, ensure_ascii=False)}\n'
        )

    blocks.append(PROMPT_FOOTER.replace("<<PHENOTYPE>>", text))
    return "\n".join(blocks)


# ------------------------------------------------------------
# Model loader
# ------------------------------------------------------------

def _load_model() -> None:
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    _model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL).to(DEVICE)
    _model.eval()


# ------------------------------------------------------------
# JSON extraction (non-greedy)
# ------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*?\}")


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _apply_field_aliases(fields_raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in fields_raw.items():
        key = _norm_str(k)
        if not key:
            continue
        mapped = FIELD_ALIASES.get(key, key)
        out[mapped] = v
    return out


# ------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------

def parse_llm(text: str) -> Dict[str, Any]:
    original = text or ""
    if not original.strip():
        return {
            "parsed_fields": {},
            "source": "llm_parser",
            "raw": original,
        }

    _load_model()
    assert _tokenizer is not None and _model is not None

    prompt = _build_prompt(original)
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
        )

    decoded = _tokenizer.decode(output[0], skip_special_tokens=True)

    if DEBUG_LLM:
        print("=== LLM RAW OUTPUT ===")
        print(decoded)
        print("======================")

    parsed_obj = _extract_first_json_object(decoded)
    fields_raw = parsed_obj.get("parsed_fields", {}) if isinstance(parsed_obj, dict) else {}
    if not isinstance(fields_raw, dict):
        fields_raw = {}

    fields_raw = _apply_field_aliases(fields_raw)

    cleaned: Dict[str, str] = {}

    for field in ALL_FIELDS:
        if field not in fields_raw:
            continue

        raw_val = fields_raw[field]

        if field == "Gram Stain":
            cleaned[field] = _normalise_gram(raw_val)
        elif field in PNV_FIELDS:
            cleaned[field] = _normalise_pnv_value(raw_val)
        else:
            cleaned[field] = _norm_str(raw_val) or "Unknown"

    cleaned = _merge_ornithine_variants(cleaned)
    cleaned = _apply_global_sugar_logic(cleaned, original)

    return {
        "parsed_fields": cleaned,
        "source": "llm_parser",
        "raw": original,
    }