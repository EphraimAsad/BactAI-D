# engine/extended_reasoner.py
# ------------------------------------------------------------
# Compute per-genus likelihoods from extended tests using signals_catalog.json

import json, os, math
from typing import Dict, List, Tuple

SIGNALS_PATH = os.path.join("data", "signals_catalog.json")
PNV = ("Positive", "Negative", "Variable")

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return default

def _log(x: float) -> float:
    # guard tiny values
    return math.log(max(x, 1e-12))

def score_genera_from_extended(parsed_ext: Dict[str, str], alpha: float = 1.0) -> Tuple[List[Tuple[str, float]], str]:
    """
    parsed_ext: dict of {ExtendedTestName: 'Positive'|'Negative'|'Variable'}
    alpha: Laplace smoothing factor
    Returns: ([(genus, score)], explanation_str)
    """
    signals = _load_json(SIGNALS_PATH, {})
    if not parsed_ext or not signals:
        return [], "No extended tests or signals available."

    # collect all genera
    genera = list(signals.keys())
    if not genera:
        return [], "No genera in signals catalog."

    # For each genus, accumulate log-likelihoods over provided tests
    scores: Dict[str, float] = {g: 0.0 for g in genera}
    contributions: Dict[str, List[str]] = {g: [] for g in genera}

    for test, val in parsed_ext.items():
        if val not in PNV:
            continue
        for g in genera:
            stats = signals.get(g, {}).get(test, None)
            if not stats:
                # unseen test for this genus → uniform
                denom = 3.0 * alpha
                prob = alpha / denom
            else:
                pos = stats.get("Positive", 0)
                neg = stats.get("Negative", 0)
                var = stats.get("Variable", 0)
                n = stats.get("_n", (pos + neg + var))
                if n <= 0:
                    denom = 3.0 * alpha
                    prob = alpha / denom
                else:
                    k = {"Positive": pos, "Negative": neg, "Variable": var}[val]
                    denom = n + 3.0 * alpha
                    prob = (k + alpha) / denom

            scores[g] += _log(prob)
            contributions[g].append(f"{test}={val}→{prob:.3f}")

    # normalize scores (softmax) for readability
    max_log = max(scores.values())
    exp_scores = {g: math.exp(s - max_log) for g, s in scores.items()}
    z = sum(exp_scores.values())
    final = sorted([(g, (exp_scores[g] / z) if z > 0 else 0.0) for g in genera], key=lambda x: x[1], reverse=True)

    # short explanation
    top_rows = []
    for g, sc in final[:5]:
        top_rows.append(f"{g}: {sc:.3f}  |  {'; '.join(contributions[g][:3])}")
    explain = "Extended-test likelihoods (top 5):\n" + "\n".join(top_rows) if top_rows else "No contributions."
    return final, explain
