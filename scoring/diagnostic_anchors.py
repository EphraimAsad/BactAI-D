# scoring/diagnostic_anchors.py
# ============================================================
# Diagnostic anchor overrides:
# - If the free-text description clearly contains certain
#   pathognomonic phrases, boost the corresponding genus
#   in the unified ranking.
# ============================================================

from __future__ import annotations

from typing import List, Dict, Any

# Simple v1 — can expand over time
DIAGNOSTIC_ANCHORS = {
    "Yersinia": [
        "bull’s-eye",
        "bull's eye",
        "cin agar",
        "pseudoappendicitis",
        "pseudo-appendicitis",
    ],
    "Campylobacter": [
        "hippurate",
        "darting motility",
    ],
    "Vibrio": [
        "tcbs agar",
        "thiosulfate citrate bile salts sucrose",
        "yellow colonies on tcbs",
        "rice-water stool",
        "rice water stool",
    ],
    "Proteus": [
        "swarming motility",
        "swarm across the plate",
        "burnt chocolate odor",
        "burned chocolate odour",
    ],
    "Listeria": [
        "tumbling motility",
        "cold enrichment",
        "grows at 4°c",
        "4°c enrichment",
    ],
    "Clostridioides": [
        "ccfa agar",
        "cycloserine cefoxitin fructose agar",
        "barnyard odor",
        "ground glass colonies",
    ],
}


def apply_diagnostic_overrides(
    description_text: str,
    unified_ranking: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    If the input description strongly suggests a particular genus
    (anchor phrases), boost that genus in the unified ranking.

    Strategy:
      - If any anchor phrase for a genus is present in the text,
        ensure that genus has at least 0.70 combined_score
        (70% overall) *if it already appears*.
      - Then re-sort by combined_score.

    This is conservative: it won't hallucinate genera that aren't
    already in the top list, but strengthens strong clinical signals.
    """
    if not description_text or not unified_ranking:
        return unified_ranking

    text_lc = description_text.lower()

    # Which genera have anchors present?
    boosted_genera = set()
    for genus, phrases in DIAGNOSTIC_ANCHORS.items():
        for p in phrases:
            if p.lower() in text_lc:
                boosted_genera.add(genus)
                break

    if not boosted_genera:
        return unified_ranking

    # Apply boost only if genus already present
    for item in unified_ranking:
        g = item.get("genus")
        if g in boosted_genera:
            score = float(item.get("combined_score", 0.0))
            if score < 0.70:
                item["combined_score"] = 0.70
                item["combined_percent"] = 70.0

    unified_ranking.sort(key=lambda d: d.get("combined_score", 0.0), reverse=True)
    return unified_ranking
