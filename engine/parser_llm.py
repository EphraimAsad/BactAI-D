# engine/parser_llm.py
# ------------------------------------------------------------
# Local LLM parser for BactAI-D (T5 fine-tune, CPU-friendly)
#
# UPDATED (EphBactAID integration):
# - Default model now points to your HF fine-tune: EphAsad/EphBactAID
# - Few-shot disabled by default (your fine-tune no longer needs it)
# - Robust output parsing:
#     * Supports JSON output (legacy)
#     * Supports "Key: Value" pairs output (your fine-tune style)
# - Merge guard (optional): LLM fills ONLY missing/Unknown fields
# - Validation/normalisation kept (PNV/Gram, sugar logic, aliases, ornithine sync)
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, Any, List, Optional
from frontend.src.model_registry import MODELS


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

# ✅ Few-shot OFF by default now (fine-tune doesn't need it)
MAX_FEWSHOT_EXAMPLES = int(os.getenv("BACTAI_LLM_FEWSHOT", "0"))

MAX_NEW_TOKENS = int(os.getenv("BACTAI_LLM_MAX_NEW_TOKENS", "256"))

DEBUG_LLM = os.getenv("BACTAI_LLM_DEBUG", "0").strip().lower() in {
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
        "TSI Pattern",
        "Colony Pattern",
        "Motility Type",
        "Odor",
        "Pigment",
        "Gas Production",
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

    # common variants from outputs
    "Voges–Proskauer Test": "VP",
    "Voges-Proskauer Test": "VP",
    "Voges–Proskauer": "VP",
    "Voges-Proskauer": "VP",
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

    # positive
    if any(x in s for x in {"positive", "pos", "+", "yes", "present", "detected", "reactive"}):
        return "Positive"

    # negative
    if any(x in s for x in {"negative", "neg", "-", "no", "none", "absent", "not detected", "no growth"}):
        return "Negative"

    # variable
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
# Gold examples (kept for backwards compat; now optional)
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
# Prompt (supports both JSON + KV outputs; fine-tune usually KV)
# ------------------------------------------------------------

PROMPT_HEADER = """
You are a microbiology phenotype parser.

Task:
- Extract ONLY explicitly stated results from the input text.
- Do NOT invent results.
- If not stated, omit the field or use "Unknown".

Output format:
- Prefer "Field: Value" lines, one per line.
- You may also output JSON if instructed.

Use the exact schema keys where possible.
"""

PROMPT_FOOTER = """
Input:
\"\"\"<<PHENOTYPE>>\"\"\"

Output:
"""


def _build_prompt(text: str) -> str:
    # Few-shot disabled by default; but we keep the capability for testing.
    blocks: List[str] = [PROMPT_HEADER]

    if MAX_FEWSHOT_EXAMPLES > 0:
        examples = _load_gold_examples()
        n = min(MAX_FEWSHOT_EXAMPLES, len(examples))
        sampled = random.sample(examples, n) if n > 0 else []
        for ex in sampled:
            inp = _norm_str(ex.get("input", ""))
            exp = ex.get("expected", {})
            if not isinstance(exp, dict):
                exp = {}
            # Show KV style to match your fine-tune
            kv_lines = "\n".join([f"{k}: {v}" for k, v in exp.items()])
            blocks.append(f'Example Input:\n"""{inp}"""\nExample Output:\n{kv_lines}\n')

    blocks.append(PROMPT_FOOTER.replace("<<PHENOTYPE>>", text))
    return "\n".join(blocks)


# ------------------------------------------------------------
# Model loader
# ------------------------------------------------------------

def _load_model() -> None:
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return

    info = MODELS["bactaid"]

    try:
        # Try online first
        if DEBUG_LLM:
            print(f"[LLM] Loading online model: {info['hf_id']}")

        _tokenizer = AutoTokenizer.from_pretrained(info["hf_id"])
        _model = AutoModelForSeq2SeqLM.from_pretrained(info["hf_id"])

    except Exception as e:
        if DEBUG_LLM:
            print(f"[LLM] Online load failed: {e}")
            print(f"[LLM] Falling back to local model: {info['local']}")

        # Force offline mode after failure
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        _tokenizer = AutoTokenizer.from_pretrained(info["local"])
        _model = AutoModelForSeq2SeqLM.from_pretrained(info["local"])

    _model = _model.to(DEVICE)
    _model.eval()



# ------------------------------------------------------------
# Output parsing helpers (JSON + KV)
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


# Match "Key: Value" (including keys with symbols like >=6%)
_KV_LINE_RE = re.compile(r"^\s*([^:\n]{2,120})\s*:\s*(.*?)\s*$")


def _extract_kv_pairs(text: str) -> Dict[str, Any]:
    """
    Parse outputs like:
      Gram Stain: Positive
      Shape: Cocci
      ...
    """
    out: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _KV_LINE_RE.match(line)
        if not m:
            continue
        k = _norm_str(m.group(1))
        v = _norm_str(m.group(2))
        if not k:
            continue
        out[k] = v
    return out


def _apply_field_aliases(fields_raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in fields_raw.items():
        key = _norm_str(k)
        if not key:
            continue
        mapped = FIELD_ALIASES.get(key, key)
        out[mapped] = v
    return out


def _clean_and_normalise(fields_raw: Dict[str, Any], original_text: str) -> Dict[str, str]:
    """
    Keep only allowed fields and normalise values into your contract.
    """
    cleaned: Dict[str, str] = {}

    # Only accept keys that match schema (or aliases already applied)
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
    cleaned = _apply_global_sugar_logic(cleaned, original_text)
    return cleaned


def _merge_guard_fill_only_missing(
    llm_fields: Dict[str, str],
    existing_fields: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Merge guard:
      - If an existing field is present and not Unknown -> do NOT overwrite.
      - If existing is missing/Unknown -> allow llm value (if not Unknown).
    """
    if not existing_fields or not isinstance(existing_fields, dict):
        return llm_fields

    out = dict(existing_fields)  # start with existing
    for k, v in llm_fields.items():
        if k not in ALL_FIELDS:
            continue
        existing_val = _norm_str(out.get(k, ""))
        existing_norm = _normalise_pnv_value(existing_val) if k in PNV_FIELDS else existing_val

        # Treat empty/Unknown as fillable
        fillable = (not existing_val) or (existing_val == "Unknown") or (existing_norm == "Unknown")
        if not fillable:
            continue

        # Only fill if LLM has something meaningful
        if _norm_str(v) and v != "Unknown":
            out[k] = v

    # Ensure we return only schema keys and strings
    final: Dict[str, str] = {}
    for k, v in out.items():
        if k in ALL_FIELDS:
            final[k] = _norm_str(v) or "Unknown"
    return final


# ------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------

def parse_llm(text: str, existing_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse phenotype text using local seq2seq model.

    Parameters
    ----------
    text : str
        phenotype description

    existing_fields : dict | None
        Optional pre-parsed fields (e.g., from rules/ext).
        If provided, LLM will ONLY fill missing/Unknown fields.

    Returns
    -------
    dict:
      {
        "parsed_fields": { ... },
        "source": "llm_parser",
        "raw": <original text>,
        "decoded": <model output> (only when DEBUG on)
      }
    """
    original = text or ""
    if not original.strip():
        return {
            "parsed_fields": (existing_fields or {}) if isinstance(existing_fields, dict) else {},
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
        print("=== LLM PROMPT (truncated) ===")
        print(prompt[:1500] + ("..." if len(prompt) > 1500 else ""))
        print("=== LLM RAW OUTPUT ===")
        print(decoded)
        print("======================")

    # 1) Try JSON extraction (legacy)
    parsed_obj = _extract_first_json_object(decoded)
    fields_raw = {}

    if isinstance(parsed_obj, dict) and parsed_obj:
        if "parsed_fields" in parsed_obj and isinstance(parsed_obj.get("parsed_fields"), dict):
            fields_raw = dict(parsed_obj["parsed_fields"])
        else:
            # in case model returned a flat JSON dict
            fields_raw = dict(parsed_obj)

    # 2) Fallback to KV parsing (your fine-tune style)
    if not fields_raw:
        fields_raw = _extract_kv_pairs(decoded)

    # 3) Alias map + normalise
    fields_raw = _apply_field_aliases(fields_raw)
    cleaned = _clean_and_normalise(fields_raw, original)

    # 4) Merge guard (optional) - fill only missing/Unknown
    if existing_fields is not None:
        cleaned = _merge_guard_fill_only_missing(cleaned, existing_fields)

    out = {
        "parsed_fields": cleaned,
        "source": "llm_parser",
        "raw": original,
    }
    if DEBUG_LLM:
        out["decoded"] = decoded
    return out