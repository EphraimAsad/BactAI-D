# rag/species_scorer.py
# ============================================================
# Species evidence scorer (deterministic, explainable)
#
# Given:
#   - target_genus
#   - parsed_fields (from fusion)
# It loads species JSON files under:
#   data/rag/knowledge_base/<Genus>/*.json  (excluding genus.json)
#
# And returns:
#   - ranked species list with scores
#   - explicit matches / conflicts
#   - marker hits (importance-weighted)
#
# Notes:
# - This is NOT an LLM. No speculation.
# - Handles list-like fields (Media / Colony Morphology) as overlap scores.
# - Handles P/N/V/Unknown fields.
# ============================================================

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional


KB_ROOT = os.path.join("data", "rag", "knowledge_base")

UNKNOWN = "Unknown"

LIST_FIELDS = {
    "Media Grown On",
    "Colony Morphology",
}

# Importance → weight
MARKER_WEIGHT = {
    "high": 3.0,
    "medium": 2.0,
    "low": 1.5,
}

# Base scoring weights
FIELD_MATCH_WEIGHT = 1.0
FIELD_CONFLICT_PENALTY = 1.2   # conflicts hurt slightly more than matches help
VARIABLE_MATCH_BONUS = 0.2     # weak support if expected is Variable


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _norm_val(v: Any) -> str:
    s = _norm_str(v)
    return s if s else UNKNOWN


def _split_semicolon(s: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"[;,\n]+", s or "") if p.strip()]
    # normalize case lightly for matching
    return [p.lower() for p in parts]


def _as_list_lower(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip().lower() for x in v if str(x).strip()]
    # string fallback
    return _split_semicolon(str(v))


def _overlap_score(expected_list: List[str], observed_list: List[str]) -> float:
    """
    Jaccard-like overlap, but anchored to expected:
      score = (# of expected items found) / (# expected)
    """
    if not expected_list:
        return 0.0
    if not observed_list:
        return 0.0
    exp = set(expected_list)
    obs = set(observed_list)
    hit = len(exp.intersection(obs))
    return hit / max(1, len(exp))


def _load_species_docs_for_genus(target_genus: str) -> List[Dict[str, Any]]:
    genus = (target_genus or "").strip()
    if not genus:
        return []

    genus_dir = os.path.join(KB_ROOT, genus)
    if not os.path.isdir(genus_dir):
        return []

    docs: List[Dict[str, Any]] = []
    for fname in sorted(os.listdir(genus_dir)):
        if not fname.lower().endswith(".json"):
            continue
        if fname == "genus.json":
            continue

        path = os.path.join(genus_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            if isinstance(doc, dict) and doc.get("level") == "species":
                doc["_source_path"] = os.path.relpath(path)
                docs.append(doc)
        except Exception:
            continue

    return docs


def _score_expected_fields(
    expected_fields: Dict[str, Any],
    parsed_fields: Dict[str, str],
) -> Tuple[float, float, List[str], List[str]]:
    """
    Returns:
      (score, possible, matches, conflicts)
    """
    score = 0.0
    possible = 0.0
    matches: List[str] = []
    conflicts: List[str] = []

    for field, expected in (expected_fields or {}).items():
        exp_norm = expected
        obs_norm = parsed_fields.get(field, UNKNOWN)

        # Skip unknown observed
        if obs_norm == UNKNOWN:
            continue

        # List fields: overlap
        if field in LIST_FIELDS:
            exp_list = _as_list_lower(exp_norm)
            obs_list = _as_list_lower(obs_norm)
            if not exp_list:
                continue

            possible += FIELD_MATCH_WEIGHT
            ov = _overlap_score(exp_list, obs_list)

            # thresholding: any overlap = support; none = conflict
            if ov > 0:
                score += FIELD_MATCH_WEIGHT * ov
                matches.append(f"{field}: overlap {ov:.2f}")
            else:
                score -= FIELD_CONFLICT_PENALTY
                conflicts.append(f"{field}: expected {expected}, got {obs_norm}")
            continue

        exp_val = _norm_val(exp_norm)
        obs_val = _norm_val(obs_norm)

        # If expected is Unknown, skip
        if exp_val == UNKNOWN:
            continue

        # If expected is Variable, weakly supportive if observed is known
        if exp_val == "Variable":
            possible += VARIABLE_MATCH_BONUS
            score += VARIABLE_MATCH_BONUS
            matches.append(f"{field}: expected Variable (observed {obs_val})")
            continue

        # Normal exact match
        possible += FIELD_MATCH_WEIGHT
        if obs_val == exp_val:
            score += FIELD_MATCH_WEIGHT
            matches.append(f"{field}: {obs_val}")
        else:
            score -= FIELD_CONFLICT_PENALTY
            conflicts.append(f"{field}: expected {exp_val}, got {obs_val}")

    return score, possible, matches, conflicts


def _score_species_markers(
    markers: List[Dict[str, Any]],
    parsed_fields: Dict[str, str],
) -> Tuple[float, float, List[str], List[str]]:
    """
    Weighted marker hits. Markers are higher-signal than generic expected fields.

    Returns:
      (score, possible, marker_hits, marker_misses)
    """
    score = 0.0
    possible = 0.0
    hits: List[str] = []
    misses: List[str] = []

    for m in markers or []:
        field = _norm_str(m.get("field"))
        val = _norm_val(m.get("value"))
        importance = _norm_str(m.get("importance")).lower() or "medium"
        w = MARKER_WEIGHT.get(importance, 2.0)

        if not field or val == UNKNOWN:
            continue

        obs = _norm_val(parsed_fields.get(field, UNKNOWN))
        if obs == UNKNOWN:
            continue

        possible += w
        if obs == val:
            score += w
            hits.append(f"{field}: {obs} ({importance})")
        else:
            score -= w * 1.1  # marker conflicts hurt more
            misses.append(f"{field}: expected {val}, got {obs} ({importance})")

    return score, possible, hits, misses


def _to_confidence(raw_score: float, possible: float) -> float:
    """
    Convert raw score into 0..1 confidence.

    We use a bounded transform:
      - normalize by possible
      - clamp into [0,1]
    """
    if possible <= 0:
        return 0.0

    # raw_score can be negative; convert to a 0..1 scale
    # normalized_score around 0 means mixed evidence
    normalized = raw_score / possible  # roughly -something .. +1
    conf = (normalized + 1.0) / 2.0    # map [-1, +1] -> [0,1] (approx)
    if conf < 0:
        conf = 0.0
    if conf > 1:
        conf = 1.0
    return float(conf)


def score_species_for_genus(
    target_genus: str,
    parsed_fields: Dict[str, str],
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Main entrypoint.

    Returns:
      {
        "genus": "...",
        "ranked": [
          {
            "species": "cloacae",
            "full_name": "Enterobacter cloacae",
            "score": 0.87,
            "raw_score": ...,
            "possible": ...,
            "matches": [...],
            "conflicts": [...],
            "marker_hits": [...],
            "marker_conflicts": [...],
            "source_file": "data/rag/knowledge_base/Enterobacter/cloacae.json"
          }, ...
        ]
      }
    """
    docs = _load_species_docs_for_genus(target_genus)
    if not docs:
        return {"genus": target_genus, "ranked": []}

    ranked: List[Dict[str, Any]] = []

    for doc in docs:
        genus = _norm_str(doc.get("genus") or target_genus)
        species = _norm_str(doc.get("species"))
        full_name = f"{genus} {species}".strip()

        expected_fields = doc.get("expected_fields") or {}
        markers = doc.get("species_markers") or []

        s1, p1, matches, conflicts = _score_expected_fields(expected_fields, parsed_fields)
        s2, p2, marker_hits, marker_conflicts = _score_species_markers(markers, parsed_fields)

        raw_score = s1 + s2
        possible = p1 + p2

        conf = _to_confidence(raw_score, possible)

        ranked.append(
            {
                "species": species or os.path.splitext(os.path.basename(doc.get("_source_path", "")))[0],
                "full_name": full_name,
                "score": conf,
                "raw_score": raw_score,
                "possible": possible,
                "matches": matches,
                "conflicts": conflicts,
                "marker_hits": marker_hits,
                "marker_conflicts": marker_conflicts,
                "source_file": doc.get("_source_path", ""),
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return {"genus": target_genus, "ranked": ranked[: max(1, int(top_n))]}