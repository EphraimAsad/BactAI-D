# backend/app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os

# FIX FOR FLASK RELOADER - Set the correct path when running from backend/
if os.path.basename(os.getcwd()) == 'backend':
    os.chdir('..')

# Get the absolute path to the project root (parent of backend folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Change working directory to project root so relative paths work
os.chdir(PROJECT_ROOT)

print(f"[DEBUG] Project root: {PROJECT_ROOT}")
print(f"[DEBUG] Current working directory: {os.getcwd()}")
print(f"[DEBUG] Looking for database at: {os.path.join(os.getcwd(), 'data', 'bacteria_db.xlsx')}")

from datetime import datetime
import pandas as pd

# Import your existing engine modules
from engine.bacteria_identifier import BacteriaIdentifier
from engine.parser_fusion import parse_text_fused

# Import ML predictor
try:
    from engine.genus_predictor import predict_genus_from_fused
    HAS_GENUS_ML = True
except Exception as e:
    print(f"[WARN] ML predictor unavailable: {e}")
    HAS_GENUS_ML = False

# Import training modules
try:
    from training.parser_eval import run_parser_eval
    HAS_PARSER_EVAL = True
except Exception as e:
    print(f"[WARN] parser_eval unavailable: {e}")
    HAS_PARSER_EVAL = False

try:
    from training.gold_trainer import train_from_gold
    HAS_GOLD_TRAINER = True
except Exception as e:
    print(f"[WARN] gold_trainer unavailable: {e}")
    HAS_GOLD_TRAINER = False

try:
    from training.field_weight_trainer import train_field_weights
    HAS_FIELD_WEIGHT_TRAINER = True
except Exception as e:
    print(f"[WARN] field_weight_trainer unavailable: {e}")
    HAS_FIELD_WEIGHT_TRAINER = False

try:
    from engine.train_genus_model import train_genus_model
    HAS_GENUS_TRAINER = True
except Exception as e:
    print(f"[WARN] genus trainer unavailable: {e}")
    HAS_GENUS_TRAINER = False

try:
    from training.rag_index_builder import build_rag_index
    HAS_RAG_INDEX_BUILDER = True
except Exception as e:
    print(f"[WARN] rag_index_builder unavailable: {e}")
    HAS_RAG_INDEX_BUILDER = False

try:
    from training.hf_sync import push_to_hf
    HAS_HF_SYNC = True
except Exception as e:
    print(f"[WARN] hf_sync unavailable: {e}")
    HAS_HF_SYNC = False

# Import scoring modules
from scoring.overall_ranker import compute_overall_scores
from scoring.diagnostic_anchors import apply_diagnostic_overrides

# Import RAG modules
from rag.rag_retriever import retrieve_rag_context
from rag.rag_generator import generate_genus_rag_explanation, get_ollama_status
from rag.species_scorer import score_species_for_genus

app = Flask(__name__)
CORS(app)

# ============================================================
# FRONTEND STATIC FILE SERVING
# ============================================================

FRONTEND_DIST = os.path.join(os.getcwd(), "frontend", "dist")
HAS_FRONTEND = os.path.exists(os.path.join(FRONTEND_DIST, "index.html"))

if HAS_FRONTEND:
    print(f"[INFO] Serving frontend from: {FRONTEND_DIST}")
else:
    print(f"[WARN] Frontend not found at: {FRONTEND_DIST}")
    print(f"[WARN] Run 'cd frontend && npm run build' to build the frontend")

@app.route('/')
def serve_index():
    """Serve the main index.html."""
    if HAS_FRONTEND:
        return send_from_directory(FRONTEND_DIST, 'index.html')
    return jsonify({
        "error": "Frontend not built",
        "message": "Run 'cd frontend && npm install && npm run build' to build the frontend",
        "api_docs": {
            "health": "/api/health",
            "identify": "POST /api/identify",
            "ollama_status": "/api/ollama-status",
        }
    }), 404

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from frontend/dist."""
    if HAS_FRONTEND:
        # Check if file exists in dist
        file_path = os.path.join(FRONTEND_DIST, path)
        if os.path.isfile(file_path):
            return send_from_directory(FRONTEND_DIST, path)
        # For SPA routing, return index.html for non-file paths
        if not path.startswith('api/'):
            return send_from_directory(FRONTEND_DIST, 'index.html')
    return jsonify({"error": "Not found"}), 404

# Load database
def load_db():
    primary = os.path.join("data", "bacteria_db.xlsx")
    fallback = "bacteria_db.xlsx"
    
    print(f"[DEBUG] Checking primary path: {os.path.abspath(primary)}")
    print(f"[DEBUG] Primary exists: {os.path.exists(primary)}")
    
    if os.path.exists(primary):
        path = primary
        print(f"[INFO] Using database at: {os.path.abspath(path)}")
    elif os.path.exists(fallback):
        path = fallback
        print(f"[INFO] Using database at: {os.path.abspath(path)}")
    else:
        # List files in current directory for debugging
        print(f"[ERROR] Database not found. Contents of current directory:")
        print(os.listdir('.'))
        if os.path.exists('data'):
            print(f"[ERROR] Contents of data directory:")
            print(os.listdir('data'))
        raise FileNotFoundError(
            f"bacteria_db.xlsx not found.\n"
            f"Looked in:\n"
            f"  - {os.path.abspath(primary)}\n"
            f"  - {os.path.abspath(fallback)}\n"
            f"Current directory: {os.getcwd()}"
        )
    
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    mtime = os.path.getmtime(path)
    return df, datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

DB, DB_LAST_UPDATED = load_db()
ENG = BacteriaIdentifier(DB)

print(f"[INFO] Database loaded successfully. Last updated: {DB_LAST_UPDATED}")
print(f"[INFO] Database has {len(DB)} records")

# Helper functions
def _confidence_band_local(p: float) -> str:
    if p >= 0.90:
        return "Excellent Identification"
    if p >= 0.80:
        return "Good Identification"
    if p >= 0.65:
        return "Acceptable Identification"
    return "Low Discrimination"

def _apply_top5_decision_confidence(unified_ranking):
    if not unified_ranking:
        return unified_ranking
    
    top = unified_ranking[0]
    top_score = float(top.get("combined_score", 0.0) or 0.0)
    top_band = _confidence_band_local(top_score)
    
    if top_band == "Low Discrimination":
        for item in unified_ranking:
            item["decision_band"] = "Low Discrimination"
        return unified_ranking
    
    unified_ranking[0]["decision_band"] = top_band
    for item in unified_ranking[1:]:
        item["decision_band"] = "Low Discrimination"
    return unified_ranking

def _format_odds_human_friendly(odds_1000: int) -> str:
    try:
        o = int(odds_1000)
    except Exception:
        o = 0
    
    if o <= 0:
        return "—"
    x = int(round(1000.0 / float(o)))
    if x <= 1:
        return "1 in 1"
    return f"1 in {x}"

def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def compute_trifusion_and_ml(text: str, use_llm_parser: bool = False):
    text = text or ""
    if not text.strip():
        return {
            "error": "Please enter a description.",
            "fused_fields": {},
            "tri_fusion_results": [],
            "ml_genus_results": [],
            "unified_ranking": [],
            "overall_scores": {},
        }

    # 1) Tri-Fusion
    try:
        fusion = parse_text_fused(text, use_llm=use_llm_parser)
    except TypeError:
        fusion = parse_text_fused(text)

    fused_fields = fusion.get("fused_fields", {})
    results = ENG.identify(fused_fields)

    # 2) ML GENUS MODEL
    ml_results_raw = []
    if HAS_GENUS_ML:
        try:
            preds = predict_genus_from_fused(fused_fields, top_k=10)
            if preds:
                for genus, prob, band in preds:
                    perc = prob * 100.0
                    ml_results_raw.append({
                        "genus": genus,
                        "probability": prob,
                        "probability_percent": perc,
                        "confidence_band": band,
                    })
        except Exception as e:
            print(f"[ERROR] ML genus prediction failed: {e}")

    # 3) UNIFIED HYBRID RANKING
    tri_blended_by_genus = {}
    for r in results:
        g = str(r.genus)
        s = (r.blended_confidence_percent() or 0.0) / 100.0
        if s > tri_blended_by_genus.get(g, 0.0):
            tri_blended_by_genus[g] = s

    ml_by_genus = {
        item["genus"]: float(item["probability"]) for item in ml_results_raw
    }

    all_genera = set(tri_blended_by_genus.keys()) | set(ml_by_genus.keys())
    unified_ranking = []

    if all_genera:
        for g in all_genera:
            tf = tri_blended_by_genus.get(g, 0.0)
            ml = ml_by_genus.get(g, 0.0)

            if ml <= 0.01:
                combined = 0.01 * tf + 0.99 * ml
            elif ml >= 0.90:
                combined = 0.3 * tf + 0.7 * ml
            else:
                combined = 0.5 * tf + 0.5 * ml

            TF_GATE = 0.30
            if tf <= TF_GATE:
                combined = min(combined, tf)

            band = _confidence_band_local(combined)
            unified_ranking.append({
                "genus": g,
                "combined_score": combined,
                "combined_percent": combined * 100.0,
                "tri_fusion_blended_percent": tf * 100.0,
                "ml_prob_percent": ml * 100.0,
                "ml_band": band,
            })

        unified_ranking = apply_diagnostic_overrides(text, unified_ranking)
        unified_ranking.sort(key=lambda d: d.get("combined_score", 0.0), reverse=True)
        unified_ranking = _apply_top5_decision_confidence(unified_ranking)

    # 4) OVERALL RANKER
    try:
        tri_scores_map = {item["genus"]: float(item.get("combined_score", 0.0) or 0.0) for item in unified_ranking}
        overall_scores = compute_overall_scores(
            ml_scores=ml_results_raw,
            tri_scores=tri_scores_map,
            top_k=5,
        )
    except Exception as e:
        print(f"[ERROR] Overall ranker failed: {e}")
        overall_scores = {
            "error": f"overall_ranker failed: {e}",
            "overall": [],
            "normalized_share_percent": [],
            "probabilities_1000": [],
        }

    return {
        "error": None,
        "fused_fields": fused_fields,
        "tri_fusion_results": [{"genus": r.genus, "confidence": r.blended_confidence_percent()} for r in results],
        "ml_genus_results": ml_results_raw,
        "unified_ranking": unified_ranking,
        "overall_scores": overall_scores,
    }

@app.route('/api/identify', methods=['POST'])
def identify():
    data = request.json
    text = data.get('text', '')
    use_llm = data.get('use_llm', False)
    
    result = compute_trifusion_and_ml(text, use_llm_parser=use_llm)
    
    if result.get('error'):
        return jsonify(result), 400
    
    ranking = result['unified_ranking'] or []
    
    # Build top-5 table
    overall = result.get('overall_scores') or {}
    overall_list = overall.get('overall') or []
    probs_1000_list = overall.get('probabilities_1000') or []
    
    share_by_genus = {}
    odds_by_genus = {}
    
    for it in overall_list:
        if not isinstance(it, dict):
            continue
        g = str(it.get("genus") or "").strip()
        if not g:
            continue
        share = it.get("normalized_share") or it.get("share") or it.get("normalized_share_percent")
        if share is not None:
            s = _safe_float(share)
            if s > 1.0:
                s = s / 100.0
            share_by_genus[g] = max(0.0, min(1.0, s))
    
    for it in probs_1000_list:
        if not isinstance(it, dict):
            continue
        g = str(it.get("genus") or "").strip()
        if not g:
            continue
        o = it.get("odds_1000") or it.get("prob_1000")
        if isinstance(o, (int, float)):
            odds_by_genus[g] = int(round(o))
    
    if not share_by_genus:
        total = sum(float(item.get("combined_score", 0.0) or 0.0) for item in ranking[:5]) or 1.0
        for item in ranking[:5]:
            genus = str(item.get("genus") or "").strip()
            if genus:
                share_by_genus[genus] = float(item.get("combined_score", 0.0) or 0.0) / total
    
    top5_rows = []
    top1_band = ranking[0].get("decision_band") if ranking else "Low Discrimination"
    
    for idx, item in enumerate(ranking[:5], start=1):
        genus = str(item.get("genus") or "").strip()
        share = share_by_genus.get(genus, 0.0)
        odds_1000 = odds_by_genus.get(genus, int(round(share * 1000)))
        prob_pct = f"{share * 100.0:.2f}%"
        odds_text = _format_odds_human_friendly(odds_1000)
        
        if top1_band == "Low Discrimination":
            confidence = "Low Discrimination"
        else:
            confidence = top1_band if idx == 1 else "Low Discrimination"
        
        top5_rows.append([genus, prob_pct, odds_text, confidence])
    
    # RAG explanations for top genus
    genus_cards = []
    if ranking:
        for item in ranking[:5]:
            genus = item["genus"]
            rag_text = None
            
            try:
                ctx = retrieve_rag_context(
                    phenotype_text=text,
                    target_genus=genus,
                    top_k=5,
                    parsed_fields=result["fused_fields"],
                )
                
                explanation = generate_genus_rag_explanation(
                    phenotype_text=text,
                    rag_context=ctx.get("llm_context_shaped", "") or ctx.get("llm_context", ""),
                    genus=genus,
                )
                
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
                except Exception as e:
                    explanation += "\n\n**Species Best Match:** Not specified"
                
                rag_text = explanation
            except Exception as e:
                print(f"[ERROR] RAG generation failed for {genus}: {e}")
                rag_text = f"(RAG error: {e})"
            
            genus_cards.append({
                "genus": genus,
                "combined_percent": item["combined_percent"],
                "tri_fusion_blended_percent": item["tri_fusion_blended_percent"],
                "ml_prob_percent": item["ml_prob_percent"],
                "decision_band": item.get("decision_band") or item.get("ml_band") or "Low Discrimination",
                "rag_text": rag_text,
            })
    
    return jsonify({
        "top5_table": top5_rows,
        "genus_cards": genus_cards,
        "debug": result
    })

@app.route('/api/evaluate-parsers', methods=['POST'])
def evaluate_parsers():
    if not HAS_PARSER_EVAL:
        return jsonify({"ok": False, "message": "parser_eval not available"}), 400
    result = run_parser_eval(mode="rules+extended")
    return jsonify(result)

@app.route('/api/train-gold', methods=['POST'])
def train_gold():
    if not HAS_GOLD_TRAINER:
        return jsonify({"ok": False, "message": "gold_trainer not available"}), 400
    result = train_from_gold()
    return jsonify(result)

@app.route('/api/train-weights', methods=['POST'])
def train_weights():
    if not HAS_FIELD_WEIGHT_TRAINER:
        return jsonify({"ok": False, "message": "field_weight_trainer not available"}), 400
    result = train_field_weights(include_llm=False)
    return jsonify(result)

@app.route('/api/train-genus', methods=['POST'])
def train_genus():
    if not HAS_GENUS_TRAINER:
        return jsonify({"ok": False, "message": "genus trainer not available"}), 400
    result = train_genus_model()
    return jsonify(result)

@app.route('/api/build-rag-index', methods=['POST'])
def build_rag():
    if not HAS_RAG_INDEX_BUILDER:
        return jsonify({"ok": False, "message": "rag_index_builder not available"}), 400
    result = build_rag_index()
    return jsonify(result)

@app.route('/api/commit-hf', methods=['POST'])
def commit_hf():
    if not HAS_HF_SYNC:
        return jsonify({"ok": False, "message": "hf_sync not available"}), 400
    
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
    result = push_to_hf(paths)
    return jsonify(result)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "db_last_updated": DB_LAST_UPDATED})

@app.route('/api/ollama-status', methods=['GET'])
def ollama_status():
    """Check Ollama connection and model availability."""
    status = get_ollama_status()
    return jsonify(status)

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"BactAI-D Backend Server Starting")
    print(f"{'='*60}")
    print(f"API available at: http://localhost:8000/api")
    print(f"Health check: http://localhost:8000/api/health")
    print(f"{'='*60}\n")
    app.run(debug=True, port=8000, host='127.0.0.1', use_reloader=False)
