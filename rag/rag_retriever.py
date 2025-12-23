# rag/rag_retriever.py
# ============================================================
# RAG retriever (Stage 2 – microbiology-aware)
#
# Key change (GENUS-FIRST):
# - The generator must NOT see multiple species dumps.
# - We retrieve GENUS-level records only for llm_context/llm_context_shaped.
# - Species is handled separately (deterministic species_scorer), not via LLM context.
#
# Improvements retained:
# - Source-type weighting (but genus-only for generator)
# - Genus-aware query expansion
# - Diversity enforcement (avoid duplicate sources)
# - Explicit ranking & score annotations for generator (DEBUG ONLY)
# - OPTIONAL: species evidence scoring (deterministic)
# - NEW: Context shaper (deterministic) -> resolves conflicts + emits genus-ready summary
#
# IMPORTANT:
# - We return THREE contexts:
#     1) llm_context         -> GENUS-only raw text (SAFE but unshaped)
#     2) llm_context_shaped  -> shaped, conflict-aware, generator-friendly
#     3) debug_context       -> includes RANK/SCORE/WEIGHTS (UI/logging only)
# ============================================================

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np

from rag.rag_embedder import embed_text, load_kb_index

# deterministic species evidence scorer (separate from generator context)
try:
    from rag.species_scorer import score_species_for_genus
    HAS_SPECIES_SCORER = True
except Exception:
    score_species_for_genus = None  # type: ignore
    HAS_SPECIES_SCORER = False


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# NOTE: We keep these for debug display + potential fallback modes.
SOURCE_TYPE_WEIGHTS = {
    "species": 1.15,
    "genus": 1.00,
    "table": 1.10,
    "note": 0.85,
}

MAX_CHUNKS_PER_SOURCE = 1

# Context shaping caps (keeps prompt within LLM limits)
SHAPER_MAX_CORE = 14
SHAPER_MAX_VARIABLE = 12
SHAPER_MAX_MATCHES = 14
SHAPER_MAX_CONFLICTS = 12
SHAPER_MAX_TOTAL_CHARS = 9000  # final guardrail


# ------------------------------------------------------------
# Similarity helper
# ------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity for normalized embeddings.
    Assumes both vectors are already L2-normalized.
    """
    return float(np.dot(a, b))


# ------------------------------------------------------------
# Context Shaper (deterministic)
# ------------------------------------------------------------

_TRAIT_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9 \/\-\(\)\[\]%>=<\+\.]*?)\s*:\s*(.+?)\s*$"
)

# Headers / junk lines we don't want treated as traits
_SHAPER_SKIP_PREFIXES = (
    "expected fields for species",
    "expected fields for genus",
    "reference context",
    "genus evidence primer",
)

def _norm_val(v: str) -> str:
    s = (v or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    return s

def _canon_bool(v: str) -> str:
    """
    Canonicalize common boolean-ish microbiology values.
    Conservative: no inference.
    """
    s = _norm_val(v).lower()
    if s in {"pos", "positive", "+", "reactive"}:
        return "Positive"
    if s in {"neg", "negative", "-", "nonreactive", "non-reactive"}:
        return "Negative"
    if s in {"none"}:
        return "None"
    if s in {"unknown", "not specified", "n/a", "na"}:
        return "Unknown"
    if s in {"variable"}:
        return "Variable"
    return _norm_val(v)

def _canon_trait_name(name: str) -> str:
    s = _norm_val(name)
    s_low = s.lower()
    if s_low == "ornitihine decarboxylase":
        return "Ornithine Decarboxylase"
    return s

def _extract_traits_from_text_block(text: str) -> List[Tuple[str, str]]:
    """
    Extract (trait, value) pairs from lines like:
      Trait Name: Value
    """
    pairs: List[Tuple[str, str]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if any(low.startswith(p) for p in _SHAPER_SKIP_PREFIXES):
            continue
        m = _TRAIT_LINE_RE.match(line)
        if not m:
            continue
        k = _canon_trait_name(m.group(1))
        v = _canon_bool(m.group(2))
        if not k or not v:
            continue
        pairs.append((k, v))
    return pairs

def _compare_vals(observed: str, reference: str) -> Optional[bool]:
    """
    Returns:
      True  -> match
      False -> conflict
      None  -> cannot compare (unknown/variable/empty)
    """
    o = _canon_bool(observed)
    r = _canon_bool(reference)

    if not o or o == "Unknown":
        return None
    if not r or r in {"Unknown", "Variable"}:
        return None

    if o == r:
        return True

    # Safe equivalences (very conservative)
    eq = {
        ("None", "Negative"),
        ("Negative", "None"),
    }
    if (o, r) in eq:
        return True

    return False

def shape_genus_context(
    *,
    target_genus: str,
    selected_chunks: List[Dict[str, Any]],
    parsed_fields: Optional[Dict[str, str]] = None,
) -> str:
    """
    Deterministic, GENUS-focused context shaper.

    It:
    - aggregates trait lines across retrieved GENUS chunks
    - identifies CORE traits (single consistent value across chunks)
    - identifies VARIABLE traits (multiple values across chunks)
    - if parsed_fields provided, derives:
        - phenotype-supported matches vs CORE traits
        - phenotype conflicts vs CORE traits
    - outputs a compact, reasoning-friendly block for the generator
    """
    genus = (target_genus or "").strip() or "Unknown"

    trait_values: Dict[str, List[str]] = {}

    for rec in selected_chunks or []:
        txt = (rec.get("text") or "").strip()
        if not txt:
            continue
        for k, v in _extract_traits_from_text_block(txt):
            trait_values.setdefault(k, []).append(v)

    # Reduce to unique canonical values
    trait_uniques: Dict[str, List[str]] = {}
    for k, vals in trait_values.items():
        uniq: List[str] = []
        for v in vals:
            vv = _canon_bool(v)
            if not vv:
                continue
            if vv not in uniq:
                uniq.append(vv)
        if uniq:
            trait_uniques[k] = uniq

    core_traits: List[Tuple[str, str]] = []
    variable_traits: List[Tuple[str, str]] = []

    for k, uniq in trait_uniques.items():
        if len(uniq) == 1:
            core_traits.append((k, uniq[0]))
        else:
            variable_traits.append((k, " / ".join(uniq)))

    PRIORITY = {
        "Gram Stain": 1,
        "Shape": 2,
        "Motility": 3,
        "Motility Type": 4,
        "Oxidase": 5,
        "Catalase": 6,
        "Oxygen Requirement": 7,
        "Lactose Fermentation": 8,
        "Glucose Fermentation": 9,
        "H2S": 10,
        "Indole": 11,
        "Urease": 12,
        "Citrate": 13,
        "ONPG": 14,
        "NaCl Tolerant (>=6%)": 15,
        "Media Grown On": 16,
        "Colony Morphology": 17,
    }

    def _sort_key(item: Tuple[str, str]) -> Tuple[int, str]:
        return (PRIORITY.get(item[0], 999), item[0].lower())

    core_traits.sort(key=_sort_key)
    variable_traits.sort(key=_sort_key)

    core_traits = core_traits[:SHAPER_MAX_CORE]
    variable_traits = variable_traits[:SHAPER_MAX_VARIABLE]

    matches: List[str] = []
    conflicts: List[str] = []

    if parsed_fields:
        for k, ref_v in core_traits:
            obs_v = parsed_fields.get(k)
            if obs_v is None:
                continue
            cmp = _compare_vals(obs_v, ref_v)
            if cmp is True:
                matches.append(f"- {k}: {_canon_bool(obs_v)} (matches reference: {ref_v})")
            elif cmp is False:
                conflicts.append(f"- {k}: {_canon_bool(obs_v)} (conflicts reference: {ref_v})")

    matches = matches[:SHAPER_MAX_MATCHES]
    conflicts = conflicts[:SHAPER_MAX_CONFLICTS]

    lines: List[str] = []
    lines.append(f"GENUS SUMMARY (reference-driven): {genus}")

    if core_traits:
        lines.append("\nCORE GENUS TRAITS (consistent across retrieved genus references):")
        for k, v in core_traits:
            lines.append(f"- {k}: {v}")
    else:
        lines.append("\nCORE GENUS TRAITS: Not available from retrieved context.")

    if variable_traits:
        lines.append("\nTRAITS VARIABLE ACROSS RETRIEVED GENUS REFERENCES (do not treat as contradictions):")
        for k, v in variable_traits:
            lines.append(f"- {k}: Variable ({v})")

    if parsed_fields:
        lines.append("\nPHENOTYPE SUPPORT (observed vs CORE traits):")
        if matches:
            lines.append("KEY MATCHES:")
            lines.extend(matches)
        else:
            lines.append("KEY MATCHES: Not specified.")

        if conflicts:
            lines.append("\nCONFLICTS (observed vs CORE traits):")
            lines.extend(conflicts)
        else:
            lines.append("\nCONFLICTS: Not specified.")

    shaped = "\n".join(lines).strip()

    if len(shaped) > SHAPER_MAX_TOTAL_CHARS:
        shaped = shaped[:SHAPER_MAX_TOTAL_CHARS].rstrip() + "\n... (truncated)"

    return shaped


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def retrieve_rag_context(
    phenotype_text: str,
    target_genus: str,
    top_k: int = 5,
    kb_path: str = "data/rag/index/kb_index.json",
    parsed_fields: Optional[Dict[str, str]] = None,
    species_top_n: int = 5,
    allow_species_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Retrieve the most relevant RAG chunks for a phenotype + genus.

    GENUS-FIRST behavior:
    - For LLM generator contexts, we retrieve ONLY genus-level records (level == "genus").
    - Species is handled separately via deterministic species_scorer.

    Optional:
      parsed_fields -> enables species evidence scoring + context shaping matches/conflicts.

    Returns:
      {
        "genus": target_genus,
        "chunks": [...],               # ranked chunk metadata (GENUS chunks unless fallback enabled)
        "llm_context": "....",         # GENUS raw text (no scores)
        "llm_context_shaped": "....",  # deterministic genus-friendly summary
        "debug_context": "....",       # annotated with rank/score/weights
        "species_evidence": { ... }    # optional deterministic species scoring
      }
    """

    kb = load_kb_index(kb_path)
    records = kb.get("records", [])

    if not records:
        return {
            "genus": target_genus,
            "chunks": [],
            "llm_context": "",
            "llm_context_shaped": "",
            "debug_context": "",
            "species_evidence": {"genus": target_genus, "ranked": []},
        }

    query_text = (phenotype_text or "").strip()
    if target_genus:
        query_text = f"{query_text}\nTarget genus: {target_genus}"

    q_emb = embed_text(query_text, normalize=True)
    target_genus_lc = (target_genus or "").strip().lower()

    scored_records: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Primary pass: STRICT genus-filtered + GENUS-LEVEL only
    # --------------------------------------------------------
    for rec in records:
        rec_genus = (rec.get("genus") or "").strip().lower()
        if target_genus_lc and rec_genus != target_genus_lc:
            continue

        level = (rec.get("level") or "").strip().lower()
        if level != "genus":
            continue  # GENUS-ONLY for generator context

        emb = rec.get("embedding")
        if emb is None:
            continue

        base_score = _cosine_similarity(q_emb, emb)
        weight = SOURCE_TYPE_WEIGHTS.get(level, 1.0)
        score = base_score * weight

        scored_records.append(
            {
                "id": rec.get("id"),
                "genus": rec.get("genus"),
                "species": rec.get("species"),
                "source_type": level,
                "path": rec.get("source_file"),
                "text": rec.get("text"),
                "score": float(score),
                "base_score": float(base_score),
                "type_weight": float(weight),
                "section": rec.get("section"),
                "role": rec.get("role"),
                "chunk_id": rec.get("chunk_id"),
            }
        )

    # --------------------------------------------------------
    # Fallback modes
    # --------------------------------------------------------
    if not scored_records and allow_species_fallback:
        # Emergency fallback: allow any level if no genus chunks exist.
        # This keeps your app functioning, but can reintroduce noise.
        for rec in records:
            rec_genus = (rec.get("genus") or "").strip().lower()
            if target_genus_lc and rec_genus != target_genus_lc:
                continue

            emb = rec.get("embedding")
            if emb is None:
                continue

            level = (rec.get("level") or "").strip().lower()
            base_score = _cosine_similarity(q_emb, emb)
            weight = SOURCE_TYPE_WEIGHTS.get(level, 1.0)
            score = base_score * weight

            scored_records.append(
                {
                    "id": rec.get("id"),
                    "genus": rec.get("genus"),
                    "species": rec.get("species"),
                    "source_type": level,
                    "path": rec.get("source_file"),
                    "text": rec.get("text"),
                    "score": float(score),
                    "base_score": float(base_score),
                    "type_weight": float(weight),
                    "section": rec.get("section"),
                    "role": rec.get("role"),
                    "chunk_id": rec.get("chunk_id"),
                }
            )

    # Sort by score
    scored_records.sort(key=lambda r: r["score"], reverse=True)

    # Diversity enforcement
    selected: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}

    for rec in scored_records:
        src = rec.get("path") or ""
        count = source_counts.get(src, 0)
        if count >= MAX_CHUNKS_PER_SOURCE:
            continue
        selected.append(rec)
        source_counts[src] = count + 1
        if len(selected) >= top_k:
            break

    # Build contexts
    llm_ctx_parts: List[str] = []
    debug_ctx_parts: List[str] = []

    for idx, rec in enumerate(selected, start=1):
        txt = (rec.get("text") or "").strip()
        if txt:
            llm_ctx_parts.append(txt)

        label = rec.get("genus") or "Unknown genus"
        if rec.get("species"):
            label = f"{label} {rec['species']}"

        debug_ctx_parts.append(
            f"[RANK {idx} | SCORE {rec['score']:.3f} | BASE {rec['base_score']:.3f} | "
            f"W {rec['type_weight']:.2f} | {label} — {rec.get('source_type')}]"
            + (
                f" [section={rec.get('section')} role={rec.get('role')}]"
                if rec.get("section") or rec.get("role")
                else ""
            )
            + "\n"
            + (txt or "")
        )

    llm_context = "\n\n".join(llm_ctx_parts).strip()
    debug_context = "\n\n".join(debug_ctx_parts).strip()

    llm_context_shaped = shape_genus_context(
        target_genus=target_genus,
        selected_chunks=selected,
        parsed_fields=parsed_fields,
    )

    # OPTIONAL: deterministic species evidence scoring
    species_evidence = {"genus": target_genus, "ranked": []}
    if parsed_fields and HAS_SPECIES_SCORER and score_species_for_genus is not None:
        try:
            species_evidence = score_species_for_genus(
                target_genus=target_genus,
                parsed_fields=parsed_fields,
                top_n=species_top_n,
            )
        except Exception:
            species_evidence = {"genus": target_genus, "ranked": []}

    return {
        "genus": target_genus,
        "chunks": selected,
        "llm_context": llm_context,
        "llm_context_shaped": llm_context_shaped,
        "debug_context": debug_context,
        "species_evidence": species_evidence,
    }