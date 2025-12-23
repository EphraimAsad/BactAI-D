# scoring/overall_ranker.py
# ============================================================
# Overall Ranker — Probability Normalisation Layer
#
# PURPOSE:
#   - Take already-computed combined scores (Tri-Fusion + ML)
#   - Normalize top-K into human-interpretable probabilities
#   - Provide odds per 1000 for UI display
#
# IMPORTANT:
#   - This module DOES NOT assign confidence labels
#   - Confidence logic lives in app.py (decision-band contract)
#
# OUTPUT CONTRACT (STRICT):
# {
#   "overall": [
#       {
#           "rank": int,
#           "genus": str,
#           "combined_score": float,
#           "normalized_share": float,   # 0–1, sums to 1.0
#       },
#       ...
#   ],
#   "probabilities_1000": [
#       {
#           "genus": str,
#           "odds_1000": int
#       },
#       ...
#   ]
# }
# ============================================================

from typing import Dict, List, Any


def compute_overall_scores(
    ml_scores: List[Dict[str, Any]],
    tri_scores: Dict[str, float],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Normalize already-computed combined scores into
    probability shares and odds for the Top-5 decision table.

    Parameters
    ----------
    ml_scores : list of dict
        Each dict contains at least:
          { "genus": str, "probability": float }
        (Used ONLY to determine candidate genera)

    tri_scores : dict
        Dict mapping genus -> combined_score (0–1)
        NOTE: This is already unified (Tri-Fusion + ML).

    top_k : int
        Number of top genera to return.

    Returns
    -------
    dict
        {
          "overall": [
              {
                "rank": int,
                "genus": str,
                "combined_score": float,
                "normalized_share": float
              }
          ],
          "probabilities_1000": [
              { "genus": str, "odds_1000": int }
          ]
        }
    """

    # --------------------------------------------------------
    # 1. Build candidate list
    # --------------------------------------------------------
    combined_rows: List[Dict[str, Any]] = []

    for genus, score in tri_scores.items():
        try:
            cs = float(score)
        except Exception:
            cs = 0.0

        if cs > 0:
            combined_rows.append({
                "genus": genus,
                "combined_score": cs
            })

    if not combined_rows:
        return {
            "overall": [],
            "probabilities_1000": [],
        }

    # --------------------------------------------------------
    # 2. Sort and trim to top_k
    # --------------------------------------------------------
    combined_rows.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    top = combined_rows[:top_k]

    # --------------------------------------------------------
    # 3. Normalize to probability shares (sum = 1.0)
    # --------------------------------------------------------
    total_score = sum(x["combined_score"] for x in top)

    if total_score <= 0:
        total_score = 1.0  # safety fallback

    overall: List[Dict[str, Any]] = []
    probabilities_1000: List[Dict[str, Any]] = []

    for idx, row in enumerate(top, start=1):
        share = row["combined_score"] / total_score

        # Clamp defensively
        share = max(0.0, min(1.0, share))

        odds_1000 = int(round(share * 1000))

        overall.append({
            "rank": idx,
            "genus": row["genus"],
            "combined_score": round(row["combined_score"], 6),
            "normalized_share": round(share, 6),
        })

        probabilities_1000.append({
            "genus": row["genus"],
            "odds_1000": odds_1000,
        })

    return {
        "overall": overall,
        "probabilities_1000": probabilities_1000,
    }