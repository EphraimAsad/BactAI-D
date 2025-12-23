# app.py
# ============================================================
# BactAI-D — Microbiology Identification (LLM-Toggle + RAG)
#
# - LLM parser OFF by default (safe for HF Spaces)
# - Checkbox to enable LLM parser:
#       "Enable LLM Parser (Phi-3 Mini — Only Applicable Locally)"
# - Tri-Fusion + ML hybrid identification
# - Hybrid weighting:
#       * If ML >= 0.90 → 0.3 * Tri-Fusion + 0.7 * ML
#       * Else          → 0.5 * Tri-Fusion + 0.5 * ML
# - Confidence bands:
#       <65%  → Low Discrimination
#       65–79 → Acceptable Identification
#       80–89 → Good Identification
#       ≥90   → Excellent Identification
# - RAG (Mistral-7B-Instruct) always enabled for top genera
# - Commit-to-HF kept with all key artefacts
#
# TOP-5 TABLE (DECISION AID) RULE:
#   ✅ Confidence is assigned AFTER unified scoring.
#   ✅ Only Rank #1 may be Acceptable/Good/Excellent.
#   ✅ If Rank #1 is Low Discrimination, ALL ranks are Low Discrimination.
#   ✅ Ranks #2–#5 are always Low Discrimination (even if their % is high).
#
# TOP-5 TABLE (DECISION AID) COLUMNS:
#   ✅ Genus
#   ✅ Probability % (within TOP-5, sums to 100%)
#   ✅ Probability (Odds) — human-friendly ("1 in X")
#   ✅ Confidence (decision_band logic above)
# ============================================================

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd
import gradio as gr

# ============================================================
# ENGINE IMPORTS
# ============================================================

from engine.bacteria_identifier import BacteriaIdentifier
from engine.parser_rules import parse_text_rules
from engine.parser_ext import parse_text_extended
from engine.parser_fusion import parse_text_fused

# We will *not* import parser_llm directly here.
# LLM usage is controlled via the `use_llm` flag passed into parse_text_fused

HAS_LLM = True  # Architecturally supported; UI toggle decides whether to use it.

# ============================================================
# ML GENUS PREDICTOR
# ============================================================

try:
    from engine.genus_predictor import predict_genus_from_fused
    HAS_GENUS_ML = True
except Exception as e:
    print(f"[app] ML predictor unavailable: {type(e).__name__}: {e}")
    HAS_GENUS_ML = False

# ============================================================
# TRAINING MODULES
# ============================================================

try:
    from training.parser_eval import run_parser_eval
    HAS_PARSER_EVAL = True
except Exception as e:
    print(f"[app] parser_eval unavailable: {type(e).__name__}: {e}")
    HAS_PARSER_EVAL = False

try:
    from training.gold_trainer import train_from_gold
    HAS_GOLD_TRAINER = True
except Exception as e:
    print(f"[app] gold_trainer unavailable: {type(e).__name__}: {e}")
    HAS_GOLD_TRAINER = False

try:
    from training.field_weight_trainer import train_field_weights
    HAS_FIELD_WEIGHT_TRAINER = True
except Exception as e:
    print(f"[app] field_weight_trainer unavailable: {type(e).__name__}: {e}")
    HAS_FIELD_WEIGHT_TRAINER = False

try:
    from engine.train_genus_model import train_genus_model
    HAS_GENUS_TRAINER = True
except Exception as e:
    print(f"[app] genus trainer unavailable: {type(e).__name__}: {e}")
    HAS_GENUS_TRAINER = False

# ============================================================
# RAG INDEX BUILDER
# ============================================================

try:
    from training.rag_index_builder import build_rag_index
    HAS_RAG_INDEX_BUILDER = True
except Exception as e:
    print(f"[app] rag_index_builder unavailable: {type(e).__name__}: {e}")
    HAS_RAG_INDEX_BUILDER = False

# ============================================================
# PHASE 1 — OVERALL RANKER
# ============================================================

from scoring.overall_ranker import compute_overall_scores

# ============================================================
# DIAGNOSTIC ANCHORS (OVERRIDES)
# ============================================================

from scoring.diagnostic_anchors import apply_diagnostic_overrides

# ============================================================
# RAG IMPORTS (Mistral + Retriever)
# ============================================================

from rag.rag_retriever import retrieve_rag_context
from rag.rag_generator import generate_genus_rag_explanation
from rag.species_scorer import score_species_for_genus

# ============================================================
# DATA LOADING
# ============================================================

def load_db() -> Tuple[pd.DataFrame, str]:
    primary = os.path.join("data", "bacteria_db.xlsx")
    fallback = "bacteria_db.xlsx"

    if os.path.exists(primary):
        path = primary
    elif os.path.exists(fallback):
        path = fallback
    else:
        raise FileNotFoundError(
            "bacteria_db.xlsx not found in 'data/' or project root."
        )

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    mtime = os.path.getmtime(path)
    return df, datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")


DB, DB_LAST_UPDATED = load_db()
ENG = BacteriaIdentifier(DB)

# ============================================================
# CONFIDENCE BANDS (FINAL CONTRACT)
# ============================================================

def _confidence_band_local(p: float) -> str:
    """
    Confidence band based on the FINAL contract:
      <0.65  -> Low Discrimination
      0.65-0.79 -> Acceptable Identification
      0.80-0.89 -> Good Identification
      >=0.90 -> Excellent Identification
    """
    if p >= 0.90:
        return "Excellent Identification"
    if p >= 0.80:
        return "Good Identification"
    if p >= 0.65:
        return "Acceptable Identification"
    return "Low Discrimination"


def _apply_top5_decision_confidence(unified_ranking: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    TOP-5 TABLE DECISION RULE:
      - Only rank #1 can be Acceptable/Good/Excellent.
      - If rank #1 is Low Discrimination -> ALL ranks Low Discrimination.
      - Ranks #2-#5 ALWAYS Low Discrimination.
    We store this as:
      item["decision_band"]  (for the top-5 table + UI labels if desired)
    """
    if not unified_ranking:
        return unified_ranking

    # Determine rank-1 band based on unified combined_score
    top = unified_ranking[0]
    top_score = float(top.get("combined_score", 0.0) or 0.0)
    top_band = _confidence_band_local(top_score)

    if top_band == "Low Discrimination":
        # All LD
        for item in unified_ranking:
            item["decision_band"] = "Low Discrimination"
        return unified_ranking

    # Rank1 gets its true band; everyone else forced LD
    unified_ranking[0]["decision_band"] = top_band
    for item in unified_ranking[1:]:
        item["decision_band"] = "Low Discrimination"
    return unified_ranking


def _format_odds_human_friendly(odds_1000: int) -> str:
    """
    Convert odds per 1000 into a human-friendly "1 in X".
    Example:
      odds_1000 = 500 -> 1 in 2
      odds_1000 = 333 -> 1 in 3
      odds_1000 = 125 -> 1 in 8
    """
    try:
        o = int(odds_1000)
    except Exception:
        o = 0

    if o <= 0:
        return "—"
    # 1000/o gives expected "1 in X"
    x = int(round(1000.0 / float(o)))
    if x <= 1:
        return "1 in 1"
    return f"1 in {x}"


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# CORE IDENTIFICATION PIPELINE
# ============================================================

def compute_trifusion_and_ml(text: str, use_llm_parser: bool = False) -> Dict[str, Any]:
    text = text or ""
    if not text.strip():
        return {
            "error": "Please enter a description.",
            "fused_fields": {},
            "tri_fusion_results": [],
            "tri_fusion_summary_markdown": "",
            "ml_genus_results": [],
            "ml_summary_markdown": "",
            "unified_summary_markdown": "",
            "unified_ranking": [],
            "overall_scores": {},
            "raw": {},
        }

    # 1) Tri-Fusion
    try:
        fusion = parse_text_fused(text, use_llm=use_llm_parser)
    except TypeError:
        fusion = parse_text_fused(text)

    fused_fields = fusion.get("fused_fields", {})
    results = ENG.identify(fused_fields)

    # Tri-Fusion summary
    tri_lines: List[str] = []
    if not results:
        tri_lines.append("No matches found.")
    else:
        tri_lines.append("Tri-Fusion Identification Results:\n")
        for r in results:
            blended = r.blended_confidence_percent()
            core = r.confidence_percent()
            true = r.true_confidence()
            emoji = "🟢" if blended >= 75 else "🟡" if blended >= 50 else "🔴"
            tri_lines.append(
                f"- **{r.genus}** — {emoji} {blended}% "
                f"(Core: {core}%, True: {true}%)"
            )
    tri_md = "\n".join(tri_lines)

    # 2) ML GENUS MODEL
    ml_results_raw: List[Dict[str, Any]] = []
    ml_lines: List[str] = []

    if not HAS_GENUS_ML:
        ml_lines.append("ML genus model not available.")
    else:
        try:
            preds = predict_genus_from_fused(fused_fields, top_k=10)
            if preds:
                ml_lines.append("ML Genus Model Results (XGBoost, Stage 12D):\n")
                band_emoji = {
                    "Excellent Identification": "🟢",
                    "Good Identification": "🟡",
                    "Acceptable Identification": "🟠",
                    "Low Discrimination": "🔴",
                }
                rank = 1
                for genus, prob, band in preds:
                    perc = prob * 100.0
                    emo = band_emoji.get(band, "⚪")
                    ml_lines.append(
                        f"{rank}. **{genus}** — {emo} {perc:.1f}% ({band})"
                    )
                    ml_results_raw.append(
                        {
                            "genus": genus,
                            "probability": prob,
                            "probability_percent": perc,
                            "confidence_band": band,
                        }
                    )
                    rank += 1
            else:
                ml_lines.append("ML model returned no predictions.")
        except Exception as e:
            ml_lines.append(f"ML genus model error: {type(e).__name__}: {e}")

    ml_md = "\n".join(ml_lines)

    # 3) UNIFIED HYBRID RANKING
    unified_lines: List[str] = []
    unified_ranking: List[Dict[str, Any]] = []

    tri_blended_by_genus: Dict[str, float] = {}
    for r in results:
        g = str(r.genus)
        s = (r.blended_confidence_percent() or 0.0) / 100.0
        if s > tri_blended_by_genus.get(g, 0.0):
            tri_blended_by_genus[g] = s

    ml_by_genus: Dict[str, float] = {
        item["genus"]: float(item["probability"]) for item in ml_results_raw
    }

    all_genera = set(tri_blended_by_genus.keys()) | set(ml_by_genus.keys())

    band_emoji = {
        "Excellent Identification": "🟢",
        "Good Identification": "🟡",
        "Acceptable Identification": "🟠",
        "Low Discrimination": "🔴",
    }

    if all_genera:
        # Build raw unified scores
        for g in all_genera:
            tf = tri_blended_by_genus.get(g, 0.0)
            ml = ml_by_genus.get(g, 0.0)

            if ml <= 0.01:
                combined = 0.01 * tf + 0.99 * ml
            elif ml >= 0.90:
                combined = 0.3 * tf + 0.7 * ml
            else:
                combined = 0.5 * tf + 0.5 * ml

            # TF Gate
            TF_GATE = 0.30
            if tf <= TF_GATE:
                combined = min(combined, tf)

            band = _confidence_band_local(combined)
            unified_ranking.append(
                {
                    "genus": g,
                    "combined_score": combined,
                    "combined_percent": combined * 100.0,
                    "tri_fusion_blended_percent": tf * 100.0,
                    "ml_prob_percent": ml * 100.0,
                    "ml_band": band,  # band based on combined score
                }
            )

        # Apply diagnostic anchor overrides
        unified_ranking = apply_diagnostic_overrides(text, unified_ranking)

        # Sort after overrides
        unified_ranking.sort(
            key=lambda d: d.get("combined_score", 0.0), reverse=True
        )

        # Apply TOP-5 decision confidence rule (rank1-only)
        unified_ranking = _apply_top5_decision_confidence(unified_ranking)

        # Build markdown summary
        unified_lines.append("Unified Hybrid Ranking (Tri-Fusion + ML Genus Model):\n")
        for rank, item in enumerate(unified_ranking[:10], start=1):
            g = item["genus"]
            combined = item["combined_score"]
            band = item.get("decision_band") or item.get("ml_band") or "Low Discrimination"
            emo = band_emoji.get(band, "⚪")
            tf = item["tri_fusion_blended_percent"] / 100.0
            ml = item["ml_prob_percent"] / 100.0
            unified_lines.append(
                f"{rank}. **{g}** — {emo} Combined: {combined*100:.1f}% "
                f"(Tri-Fusion: {tf*100:.1f}% | ML: {ml*100:.1f}% — {band})"
            )

    unified_md = "\n".join(unified_lines)

    # 4) OVERALL RANKER (TOP-5 NORMALISATION)
    try:
        # NOTE: keep this contract stable for now; we will refactor overall_ranker next.
        tri_scores_map = {item["genus"]: float(item.get("combined_score", 0.0) or 0.0) for item in unified_ranking}

        overall_scores = compute_overall_scores(
            ml_scores=ml_results_raw,
            tri_scores=tri_scores_map,
            top_k=5,
        )
    except Exception as e:
        overall_scores = {
            "error": f"overall_ranker failed: {type(e).__name__}: {e}",
            "overall": [],
            "normalized_share_percent": [],
            "probabilities_1000": [],
        }

    return {
        "error": None,
        "fused_fields": fused_fields,
        "tri_fusion_results": results,
        "tri_fusion_summary_markdown": tri_md,
        "ml_genus_results": ml_results_raw,
        "ml_summary_markdown": ml_md,
        "unified_summary_markdown": unified_md,
        "unified_ranking": unified_ranking,
        "overall_scores": overall_scores,
        "raw": fusion,
    }


# ============================================================
# GENUS CARD RENDERER
# ============================================================

def _genus_card_markdown(
    item: Dict[str, Any],
    rank: int,
    rag_text: str | None = None,
) -> str:
    genus = item["genus"]
    combined = item["combined_percent"]
    tf = item["tri_fusion_blended_percent"]
    ml = item["ml_prob_percent"]

    # Show the DECISION confidence band (rank1-only rule)
    decision_band = item.get("decision_band") or item.get("ml_band") or "Low Discrimination"

    if combined >= 80:
        bar_color = "#1e88e5"
    elif combined >= 65:
        bar_color = "#43a047"
    elif combined >= 50:
        bar_color = "#fb8c00"
    else:
        bar_color = "#e53935"

    bar_html = f"""
<div style="background:rgba(255,255,255,0.08); border-radius:6px; padding:4px; margin-top:4px; margin-bottom:8px;">
  <div style="height:12px; width:{combined:.1f}%; max-width:100%; background:{bar_color}; border-radius:4px;"></div>
</div>
"""

    rag_section = ""
    if rag_text:
        rag_section = f"""
#### RAG Interpretation (Genus-Level)

{rag_text}
"""

    return f"""
### Rank {rank}: **{genus}**

{bar_html}

- **Combined Score:** {combined:.1f}%
- **Tri-Fusion (Blended):** {tf:.1f}%
- **ML Probability:** {ml:.1f}%
- **Decision Confidence:** {decision_band}

{rag_section}
"""


# ============================================================
# IDENTIFICATION CALLBACK
# ============================================================

def run_identification(text: str, use_llm_parser: bool):
    result = compute_trifusion_and_ml(text, use_llm_parser=use_llm_parser)

    # DEBUG payload
    debug_payload = {
        "fused_fields": result["fused_fields"],
        "tri_fusion_summary_markdown": result["tri_fusion_summary_markdown"],
        "ml_genus_results": result["ml_genus_results"],
        "unified_summary_markdown": result["unified_summary_markdown"],
        "unified_ranking": result["unified_ranking"],
        "overall_scores": result["overall_scores"],
        "raw": result["raw"],
    }

    ranking = result["unified_ranking"] or []

    # ------------------------------------------------------------
    # Top-5 Decision Table (ROBUST, APP-SIDE)
    # ------------------------------------------------------------
    # We do NOT trust overall_ranker yet.
    # We defensively reconstruct probabilities so the table always fills.
    # ------------------------------------------------------------

    top5_rows: List[List[str]] = []

    overall = result.get("overall_scores") or {}
    overall_list = overall.get("overall") or []
    probs_1000_list = overall.get("probabilities_1000") or []

    share_by_genus: Dict[str, float] = {}
    odds_by_genus: Dict[str, int] = {}

    # 1) Normalized share
    for it in overall_list:
        if not isinstance(it, dict):
            continue
        g = str(it.get("genus") or "").strip()
        if not g:
            continue

        share = (
            it.get("normalized_share")
            or it.get("share")
            or it.get("normalized_share_percent")
        )

        if share is not None:
            s = _safe_float(share)
            if s > 1.0:  # percent → fraction
                s = s / 100.0
            share_by_genus[g] = max(0.0, min(1.0, s))

    # 2) Odds /1000
    for it in probs_1000_list:
        if not isinstance(it, dict):
            continue
        g = str(it.get("genus") or "").strip()
        if not g:
            continue
        o = it.get("odds_1000") or it.get("prob_1000")
        if isinstance(o, (int, float)):
            odds_by_genus[g] = int(round(o))

    # 3) HARD FALLBACK — derive from unified_ranking if needed
    if not share_by_genus:
        total = sum(float(item.get("combined_score", 0.0) or 0.0) for item in ranking[:5]) or 1.0
        for item in ranking[:5]:
            genus = str(item.get("genus") or "").strip()
            if genus:
                share_by_genus[genus] = float(item.get("combined_score", 0.0) or 0.0) / total

    # 4) Build table rows IN RANK ORDER
    top1_band = ranking[0].get("decision_band") if ranking else "Low Discrimination"

    for idx, item in enumerate(ranking[:5], start=1):
        genus = str(item.get("genus") or "").strip()

        share = share_by_genus.get(genus, 0.0)
        # If overall_ranker doesn't provide odds, approximate odds_1000 from share.
        odds_1000 = odds_by_genus.get(genus, int(round(share * 1000)))

        prob_pct = f"{share * 100.0:.2f}%"
        odds_text = _format_odds_human_friendly(odds_1000)

        if top1_band == "Low Discrimination":
            confidence = "Low Discrimination"
        else:
            confidence = top1_band if idx == 1 else "Low Discrimination"

        top5_rows.append([
            genus,
            prob_pct,
            odds_text,
            confidence,
        ])

    # RAG explanations for top genera (rank 1)
    rag_summaries: Dict[str, str] = {}
    if ranking:
        top_item = ranking[0]
        genus = top_item["genus"]

        try:
            ctx = retrieve_rag_context(
                phenotype_text=text,
                target_genus=genus,
                top_k=5,
                parsed_fields=result["fused_fields"],  # 🔑 enables species scoring
            )

            # 🔍 HF SPACES DEBUG LOGGING
            print("\n" + "=" * 80)
            print("RAG DEBUG — GENERATOR INPUT")
            print("=" * 80)

            print("\n[PHENOTYPE]")
            print(text)

            print("\n[LLM CONTEXT]")
            print(ctx.get("llm_context_shaped", ""))

            print("\n[DEBUG CONTEXT]")
            print(ctx.get("debug_context", ""))

            print("=" * 80 + "\n")
            # 🔍 END DEBUG

            explanation = generate_genus_rag_explanation(
                phenotype_text=text,
                rag_context=ctx.get("llm_context_shaped", "") or ctx.get("llm_context", ""),
                genus=genus,
            )

            # -------------------------------
            # SPECIES BEST MATCH
            # -------------------------------
            try:
                species_out = score_species_for_genus(
                    target_genus=genus,
                    parsed_fields=result["fused_fields"],
                    top_n=1,
                )
                ranked = species_out.get("ranked", []) if isinstance(species_out, dict) else []
                if ranked:
                    best = ranked[0]
                    full_name = str(best.get("full_name") or "").strip()
                    score = best.get("score")
                    if full_name:
                        if isinstance(score, (int, float)):
                            explanation += f"\n\n**Species Best Match:** {full_name} ({float(score) * 100.0:.1f}%)"
                        else:
                            explanation += f"\n\n**Species Best Match:** {full_name}"
                else:
                    explanation += "\n\n**Species Best Match:** Not specified"
            except Exception:
                explanation += "\n\n**Species Best Match:** Not specified"

            rag_summaries[genus] = explanation
        except Exception as e:
            rag_summaries[genus] = f"(RAG error: {type(e).__name__}: {e})"

    # Accordions
    accordion_updates = []
    markdown_updates = []
    for _ in range(5):
        accordion_updates.append(gr.update(visible=False, open=False, label=""))
        markdown_updates.append("")

    for idx, item in enumerate(ranking[:5]):
        decision_band = item.get("decision_band") or "Low Discrimination"
        label = f"{item['genus']} — {item['combined_percent']:.1f}% — {decision_band}"
        accordion_updates[idx] = gr.update(
            visible=True,
            open=(idx == 0),
            label=label,
        )
        rag_text = rag_summaries.get(item["genus"])
        markdown_updates[idx] = _genus_card_markdown(
            item,
            rank=idx + 1,
            rag_text=rag_text,
        )

    return debug_payload, top5_rows, *accordion_updates, *markdown_updates


# ============================================================
# PARSER DEBUG CALLBACKS
# ============================================================

def run_rule_parser(text: str):
    return gr.update(visible=True, open=True), parse_text_rules(text or "")

def run_extended_parser(text: str):
    return gr.update(visible=True, open=True), parse_text_extended(text or "")

def run_trifusion_debug(text: str, use_llm_parser: bool):
    result = compute_trifusion_and_ml(text or "", use_llm_parser=use_llm_parser)
    return (
        gr.update(visible=True, open=True),
        result,
        result["tri_fusion_summary_markdown"],
        result["ml_summary_markdown"],
        result["unified_summary_markdown"],
    )


# ============================================================
# TRAINING CALLBACKS
# ============================================================

def run_parser_evaluation():
    if not HAS_PARSER_EVAL:
        return gr.update(visible=True, open=True), {
            "ok": False,
            "message": "parser_eval not available.",
        }
    return gr.update(visible=True, open=True), run_parser_eval(mode="rules+extended")

def run_gold_training():
    if not HAS_GOLD_TRAINER:
        return gr.update(visible=True, open=True), {
            "ok": False,
            "message": "gold_trainer not available.",
        }
    return gr.update(visible=True, open=True), train_from_gold()

def run_field_weight_training():
    if not HAS_FIELD_WEIGHT_TRAINER:
        return gr.update(visible=True, open=True), {
            "ok": False,
            "message": "field_weight_trainer not available.",
        }
    out = train_field_weights(include_llm=False)
    return gr.update(visible=True, open=True), out

def run_genus_training():
    if not HAS_GENUS_TRAINER:
        return gr.update(visible=True, open=True), {
            "ok": False,
            "message": "genus trainer not available.",
        }
    out = train_genus_model()
    return gr.update(visible=True, open=True), out

def run_rag_index_builder():
    if not HAS_RAG_INDEX_BUILDER:
        return gr.update(visible=True, open=True), {
            "ok": False,
            "message": "rag_index_builder not available.",
        }
    out = build_rag_index()
    return gr.update(visible=True, open=True), out

def commit_to_hf():
    from training.hf_sync import push_to_hf

    paths = [
        "data/extended_schema.json",
        "data/extended_proposals.jsonl",
        "data/signals_catalog.json",
        "data/field_weights.json",
        "data/feature_schema.json",
        "models/genus_xgb.json",
        "models/genus_xgb_meta.json",
        "data/llm_gold_examples.json",
        "data/rag/index/kb_index.json",
    ]
    return push_to_hf(paths)


# ============================================================
# UI + BACKGROUND
# ============================================================

CSS = """
html, body {
    height: 100%;
}
body {
    background-image: url('static/eph.jpeg');
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container {
    background: rgba(0, 0, 0, 0.55) !important;
    backdrop-filter: blur(14px);
    border-radius: 16px !important;
}
textarea, input[type="text"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: #e5e7eb !important;
    border-radius: 10px !important;
}
button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    transition: 0.2s ease;
}
button:hover {
    background: rgba(255,255,255,0.16) !important;
    border-color: #90caf9 !important;
}
.gr-accordion {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
}
.gr-accordion:hover {
    border-color: rgba(255,255,255,0.32) !important;
}
/* Ensure expanded accordion content is not clipped */
.gr-accordion .wrap,
.gr-accordion .gr-markdown {
    max-height: none !important;
    overflow: visible !important;
}

/* Improve readability of long RAG text */
.gr-accordion .gr-markdown {
    line-height: 1.6;
    padding-bottom: 12px;
}
"""

# ============================================================
# BUILD UI
# ============================================================

def create_app():
    with gr.Blocks(
        css=CSS,
        title="BactAI-D — Microbiology Identification",
    ) as demo:

        gr.Markdown(
            f"# 🧫 BactAI-D — Microbiology Phenotype Identification\n"
            f"**Database updated:** {DB_LAST_UPDATED}\n\n"
            "Rule-based parsing, extended schema, ML genus prediction, and "
            "RAG (knowledge base + Mistral-7B-Instruct) are combined into a "
            "unified hybrid identification engine."
        )

        llm_toggle = gr.Checkbox(
            label="Enable LLM Parser (Phi-3 Mini — Only Applicable Locally)",
            value=False,
        )

        with gr.Tabs():

            # --------------------------------------------------------
            # TAB 1 — IDENTIFICATION
            # --------------------------------------------------------
            with gr.Tab("🧬 Identification"):

                text_in = gr.Textbox(
                    label="Phenotype Description",
                    lines=8,
                    placeholder="Paste your microbiology description here…",
                )

                analyse_btn = gr.Button("🔍 Analyse & Identify")

                debug_json = gr.JSON(
                    label="Debug: fused fields + ML + unified ranking + overall"
                )

                # UPDATED table (Decision Table)
                top5_table = gr.Dataframe(
                    headers=["Genus", "Probability % (Top 5)", "Probability (Odds)", "Confidence"],
                    row_count=5,
                    col_count=4,
                    interactive=False,
                    label="Top 5 Genus Predictions (Decision Table)",
                )

                genus_accordions = []
                genus_markdowns = []

                for i in range(5):
                    with gr.Accordion(
                        f"Rank {i+1}",
                        visible=False,
                        open=False,
                    ) as acc:
                        md = gr.Markdown("")
                    genus_accordions.append(acc)
                    genus_markdowns.append(md)

                analyse_btn.click(
                    fn=run_identification,
                    inputs=[text_in, llm_toggle],
                    outputs=[debug_json, top5_table, *genus_accordions, *genus_markdowns],
                )

            # --------------------------------------------------------
            # TAB 2 — PARSERS DEBUG
            # --------------------------------------------------------
            with gr.Tab("🧪 Parsers (Debug)"):

                text2 = gr.Textbox(
                    label="Microbiology description",
                    lines=6,
                    placeholder="Paste description…",
                )

                rule_btn = gr.Button("Parse (Rule Parser)")
                ext_btn = gr.Button("Parse (Extended Tests)")
                tri_btn = gr.Button("Parse & Identify (Tri-Fusion + ML)")

                with gr.Accordion("Rule Parser Output", open=False, visible=False) as rule_panel:
                    rule_json = gr.JSON()

                with gr.Accordion("Extended Parser Output", open=False, visible=False) as ext_panel:
                    ext_json = gr.JSON()

                with gr.Accordion("Tri-Fusion Debug Output", open=False, visible=False) as tri_panel:
                    tri_json = gr.JSON()
                    tri_summary = gr.Markdown()
                    tri_ml_summary = gr.Markdown()
                    tri_unified_summary = gr.Markdown()

                rule_btn.click(run_rule_parser, [text2], [rule_panel, rule_json])
                ext_btn.click(run_extended_parser, [text2], [ext_panel, ext_json])
                tri_btn.click(
                    run_trifusion_debug,
                    [text2, llm_toggle],
                    [tri_panel, tri_json, tri_summary, tri_ml_summary, tri_unified_summary],
                )

            # --------------------------------------------------------
            # TAB 3 — TRAINING
            # --------------------------------------------------------
            with gr.Tab("📚 Training & Sync"):

                gr.Markdown(
                    "Evaluate parsers, train from gold tests, tune parser weights, "
                    "train the genus-level model, build the RAG index, and commit "
                    "artefacts back to the HF Space repository."
                )

                eval_btn = gr.Button("📊 Evaluate Parsers")
                train_btn = gr.Button("🧬 Train from Gold Tests")
                weight_btn = gr.Button("⚖️ Train Parser Weights")
                genus_btn = gr.Button("🧬 Train Genus Model")
                rag_btn = gr.Button("🧱 Build RAG Index")
                commit_btn = gr.Button("⬆️ Commit to HF")

                with gr.Accordion("Parser Evaluation Summary", open=False, visible=False) as eval_panel:
                    eval_json = gr.JSON()

                with gr.Accordion("Gold Training Summary", open=False, visible=False) as train_panel:
                    train_json = gr.JSON()

                with gr.Accordion("Field Weight Training Summary", open=False, visible=False) as weight_panel:
                    weight_json = gr.JSON()

                with gr.Accordion("Genus Model Training Summary", open=False, visible=False) as genus_panel:
                    genus_json = gr.JSON()

                with gr.Accordion("RAG Index Build Summary", open=False, visible=False) as rag_panel:
                    rag_json = gr.JSON()

                commit_output = gr.JSON(label="Commit Output")

                eval_btn.click(run_parser_evaluation, [], [eval_panel, eval_json])
                train_btn.click(run_gold_training, [], [train_panel, train_json])
                weight_btn.click(run_field_weight_training, [], [weight_panel, weight_json])
                genus_btn.click(run_genus_training, [], [genus_panel, genus_json])
                rag_btn.click(run_rag_index_builder, [], [rag_panel, rag_json])
                commit_btn.click(commit_to_hf, None, commit_output)

        gr.Markdown("<br><center>Built by <b>Zain Asad</b></center><br>")

    return demo


demo = create_app()

if __name__ == "__main__":
    demo.launch()