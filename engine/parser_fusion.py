# engine/parser_fusion.py
# ------------------------------------------------------------
# Tri-Parser Fusion — Stage 12B (Weighted, SOTA-style)
#
# This module combines:
#   - Rule parser      (parser_rules.parse_text_rules)
#   - Extended parser  (parser_ext.parse_text_extended)
#   - LLM parser       (parser_llm.parse_llm)    [optional]
#
# using per-field reliability weights learned in Stage 12A
# and stored in:
#   data/field_weights.json
#
# Behaviour:
#   - For each field, gather predictions from available parsers.
#   - For that field, load weights:
#         field_weights[field]  (if present)
#         else global weights
#         else equal weights across available parsers
#   - Discard parsers that:
#         * did not predict the field
#         * or only predicted "Unknown"
#   - Group by predicted value and sum the weights of parsers
#     that voted for each value.
#   - Choose the value with highest total weight.
#     Tie-break: prefer rules > extended > llm if needed.
#
# Output format:
#   {
#     "fused_fields": { field: value, ... },   # used by DB identifier AND genus ML
#     "by_parser": {
#       "rules": { ... },
#       "extended": { ... },
#       "llm": { ... }   # may be empty
#     },
#     "votes": {
#       field_name: {
#         "per_parser": {
#           "rules": {"value": "Positive", "weight": 0.95},
#           "extended": {"value": "Unknown", "weight": 0.03},
#           ...
#         },
#         "summed": {
#           "Positive": 0.97,
#           "Negative": 0.02
#         },
#         "chosen": "Positive"
#       },
#       ...
#     },
#     "weights_meta": {
#       "has_weights_file": True/False,
#       "weights_path": "data/field_weights.json",
#       "meta": { ... }  # from file if present
#     }
#   }
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from engine.parser_rules import parse_text_rules
from engine.parser_ext import parse_text_extended

# Optional LLM parser
try:
    from engine.parser_llm import parse_llm as parse_text_llm  # type: ignore
    HAS_LLM = True
except Exception:
    parse_text_llm = None  # type: ignore
    HAS_LLM = False

# Path to learned weights
FIELD_WEIGHTS_PATH = os.path.join("data", "field_weights.json")

UNKNOWN = "Unknown"
PARSER_ORDER = ["rules", "extended", "llm"]  # tie-breaking priority


# ------------------------------------------------------------
# Weights loading and helpers
# ------------------------------------------------------------

def _load_field_weights(path: str = FIELD_WEIGHTS_PATH) -> Dict[str, Any]:
    """
    Load the JSON weights file produced by Stage 12A.

    Expected structure:
      {
        "global": { "rules": 0.7, "extended": 0.2, "llm": 0.1 },
        "fields": {
          "DNase": {
            "rules": 0.95,
            "extended": 0.03,
            "llm": 0.02,
            "support": 123
          },
          ...
        },
        "meta": { ... }
      }

    If the file is missing or broken, fall back to empty dict,
    triggering equal-weight behaviour later.
    """
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


FIELD_WEIGHTS_RAW: Dict[str, Any] = _load_field_weights()
HAS_WEIGHTS_FILE: bool = bool(FIELD_WEIGHTS_RAW)


def _normalise_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalise parser -> score into weights summing to 1.
    If all scores are zero or dict is empty, return equal weights.
    """
    cleaned = {k: max(0.0, float(v)) for k, v in scores.items()}
    total = sum(cleaned.values())

    if total <= 0:
        n = len(cleaned) or 1
        return {k: 1.0 / n for k in cleaned}

    return {k: v / total for k, v in cleaned.items()}


def _get_base_weights_for_parsers(include_llm: bool) -> Dict[str, float]:
    """
    Equal-weight distribution across available parsers.
    Used when no learned weights are available.
    """
    parsers = ["rules", "extended"]
    if include_llm:
        parsers.append("llm")

    n = len(parsers) or 1
    return {p: 1.0 / n for p in parsers}


def _get_weights_for_field(field_name: str, include_llm: bool) -> Dict[str, float]:
    """
    Get weights for a specific field.

    Priority:
      1) FIELD_WEIGHTS_RAW["fields"][field_name]
      2) FIELD_WEIGHTS_RAW["global"]
      3) Equal weights

    Always:
      - Drop 'llm' if include_llm == False
      - Normalise
    """
    if not FIELD_WEIGHTS_RAW:
        return _normalise_scores(_get_base_weights_for_parsers(include_llm))

    fields_block = FIELD_WEIGHTS_RAW.get("fields", {}) or {}
    global_block = FIELD_WEIGHTS_RAW.get("global", {}) or {}

    raw: Dict[str, float] = {}

    field_entry = fields_block.get(field_name)
    if isinstance(field_entry, dict):
        for k, v in field_entry.items():
            if k in ("rules", "extended", "llm"):
                raw[k] = float(v)

    if not raw and isinstance(global_block, dict):
        for k, v in global_block.items():
            if k in ("rules", "extended", "llm"):
                raw[k] = float(v)

    if not raw:
        raw = _get_base_weights_for_parsers(include_llm)

    if not include_llm:
        raw.pop("llm", None)

    if not raw:
        raw = _get_base_weights_for_parsers(include_llm=False)

    return _normalise_scores(raw)


# ------------------------------------------------------------
# Fusion logic
# ------------------------------------------------------------

def _clean_pred_value(val: Optional[str]) -> Optional[str]:
    """
    Treat None, empty string, or explicit "Unknown" as missing.
    """
    if val is None:
        return None

    s = str(val).strip()
    if not s:
        return None

    if s.lower() == UNKNOWN.lower():
        return None

    return s


def parse_text_fused(text: str, use_llm: Optional[bool] = None) -> Dict[str, Any]:
    """
    Main tri-parser fusion entrypoint.

    Parameters
    ----------
    text : str
    use_llm : bool or None
        True  -> include LLM
        False -> exclude LLM
        None  -> include if available

    Returns
    -------
    Dict[str, Any]
        Full fusion output including votes and per-parser breakdowns.
    """
    original = text or ""
    include_llm = HAS_LLM if use_llm is None else bool(use_llm)

    rules_out = parse_text_rules(original) or {}
    ext_out = parse_text_extended(original) or {}

    rules_fields = dict(rules_out.get("parsed_fields", {}))
    ext_fields = dict(ext_out.get("parsed_fields", {}))

    llm_fields: Dict[str, Any] = {}
    if include_llm and parse_text_llm is not None:
        try:
            llm_out = parse_text_llm(original)
            if isinstance(llm_out, dict):
                if "parsed_fields" in llm_out:
                    llm_fields = dict(llm_out.get("parsed_fields", {}))
                else:
                    llm_fields = {str(k): v for k, v in llm_out.items()}
        except Exception:
            llm_fields = {}
    else:
        include_llm = False

    by_parser: Dict[str, Dict[str, Any]] = {
        "rules": rules_fields,
        "extended": ext_fields,
        "llm": llm_fields if include_llm else {},
    }

    candidate_fields = (
        set(rules_fields.keys())
        | set(ext_fields.keys())
        | set(llm_fields.keys())
    )

    fused_fields: Dict[str, Any] = {}
    votes_debug: Dict[str, Any] = {}

    for field in sorted(candidate_fields):
        weights = _get_weights_for_field(field, include_llm)

        parser_preds: Dict[str, Optional[str]] = {
            "rules": _clean_pred_value(rules_fields.get(field)),
            "extended": _clean_pred_value(ext_fields.get(field)),
            "llm": _clean_pred_value(llm_fields.get(field)) if include_llm else None,
        }

        per_parser_info: Dict[str, Any] = {}
        value_scores: Dict[str, float] = {}

        for parser_name in PARSER_ORDER:
            if parser_name == "llm" and not include_llm:
                continue

            pred = parser_preds.get(parser_name)
            w = float(weights.get(parser_name, 0.0))

            per_parser_info[parser_name] = {
                "value": pred if pred is not None else UNKNOWN,
                "weight": w,
            }

            if pred is not None:
                value_scores[pred] = value_scores.get(pred, 0.0) + w

        if not value_scores:
            fused_value = UNKNOWN
        else:
            max_score = max(value_scores.values())
            best_values = [v for v, s in value_scores.items() if s == max_score]

            if len(best_values) == 1:
                fused_value = best_values[0]
            else:
                fused_value = best_values[0]
                for parser_name in PARSER_ORDER:
                    if parser_name == "llm" and not include_llm:
                        continue
                    if parser_preds.get(parser_name) in best_values:
                        fused_value = parser_preds[parser_name]  # type: ignore
                        break

        fused_fields[field] = fused_value
        votes_debug[field] = {
            "per_parser": per_parser_info,
            "summed": value_scores,
            "chosen": fused_value,
        }

    weights_meta = {
        "has_weights_file": HAS_WEIGHTS_FILE,
        "weights_path": FIELD_WEIGHTS_PATH,
        "meta": FIELD_WEIGHTS_RAW.get("meta", {}) if HAS_WEIGHTS_FILE else {},
    }

    return {
        "fused_fields": fused_fields,
        "by_parser": by_parser,
        "votes": votes_debug,
        "weights_meta": weights_meta,
    }