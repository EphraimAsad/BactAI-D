# training/rag_index_builder.py
# ============================================================
# Build RAG index from JSON knowledge base (SECTION-AWARE)
#
# - Walks data/rag/knowledge_base/<Genus>/
# - Reads genus.json + species JSONs
# - Converts JSON → structured SECTION records
# - Computes embeddings via rag.rag_embedder.embed_texts
# - Writes index to data/rag/index/kb_index.json
#
# Output record schema (LOCKED):
# {
#   "id": "Enterobacter|cloacae|species_markers|0",
#   "level": "genus" | "species",
#   "genus": "Enterobacter",
#   "species": "cloacae" | null,
#   "section": "...",
#   "role": "...",
#   "text": "...",
#   "source_file": "...",
#   "chunk_id": 0,
#   "embedding": [...]
# }
#
# NOTE:
# We keep the locked keys above. We MAY add extra keys (non-breaking),
# e.g. "field_key" to support future scoring/weighting.
# ============================================================

from __future__ import annotations

import json
import os
import re
from typing import Dict, Any, List, Tuple, Optional

from rag.rag_embedder import embed_texts, EMBEDDING_MODEL_NAME

KB_ROOT = os.path.join("data", "rag", "knowledge_base")
INDEX_DIR = os.path.join("data", "rag", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "kb_index.json")

# Chunk size is per-section. This should generally be smaller than the generator
# prompt chunk budget so retriever can pick "tight" context blocks.
DEFAULT_MAX_CHARS = int(os.getenv("BACTAI_RAG_CHUNK_MAX_CHARS", "1100"))

# ------------------------------------------------------------
# TEXT HELPERS
# ------------------------------------------------------------

def _norm_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""

def _safe_join(items: List[str], sep: str = " ") -> str:
    return sep.join([s for s in items if s])

def _bullet_lines(items: List[str], prefix: str = "- ") -> str:
    clean = [i.strip() for i in items if isinstance(i, str) and i.strip()]
    if not clean:
        return ""
    return "\n".join(prefix + c for c in clean)

def _title_case_field(field_name: str) -> str:
    # Keep parser field names stable (don’t “prettify” them incorrectly)
    return field_name.strip()

def _format_expected_fields(expected_fields: Dict[str, Any]) -> str:
    """
    Turn your expected_fields into a compact, self-contained key:value block.
    Handles strings, lists, and simple scalars.
    """
    if not isinstance(expected_fields, dict) or not expected_fields:
        return ""

    lines: List[str] = []
    for k in sorted(expected_fields.keys(), key=lambda s: str(s).lower()):
        key = _title_case_field(str(k))
        v = expected_fields.get(k)

        if isinstance(v, list):
            vals = [str(x).strip() for x in v if str(x).strip()]
            if vals:
                lines.append(f"{key}: " + "; ".join(vals))
            else:
                lines.append(f"{key}: Unknown")
        else:
            val = _norm_str(v) or "Unknown"
            lines.append(f"{key}: {val}")

    return "\n".join(lines)

def _as_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    if v is None:
        return []
    s = str(v).strip()
    return [s] if s else []

def _is_unknown(v: str) -> bool:
    return (v or "").strip().lower() in {"unknown", "not specified", "n/a", "na", ""}

def _expected_fields_to_sentences(
    expected_fields: Dict[str, Any],
    *,
    subject: str,
) -> str:
    """
    Convert expected_fields into DECLARATIVE microbiology statements.
    This is the key fix for "Not specified" RAG outputs:
    LLMs treat these as evidence-like assertions rather than schema metadata.
    """
    if not isinstance(expected_fields, dict) or not expected_fields:
        return ""

    # Prefer these first (front-load the most diagnostic traits)
    priority = [
        "Gram Stain",
        "Shape",
        "Oxygen Requirement",
        "Motility",
        "Motility Type",
        "Capsule",
        "Spore Formation",
        "Haemolysis",
        "Haemolysis Type",
        "Oxidase",
        "Catalase",
        "Indole",
        "Urease",
        "Citrate",
        "Methyl Red",
        "VP",
        "H2S",
        "ONPG",
        "Nitrate Reduction",
        "NaCl Tolerant (>=6%)",
        "Growth Temperature",
        "Media Grown On",
        "Colony Morphology",
        "Colony Pattern",
        "Pigment",
        "TSI Pattern",
        "Gas Production",
    ]

    # Then everything else, stable order
    all_keys = list(expected_fields.keys())
    ordered = []
    seen = set()
    for k in priority:
        if k in expected_fields:
            ordered.append(k)
            seen.add(k)
    for k in sorted(all_keys, key=lambda s: str(s).lower()):
        if k not in seen:
            ordered.append(k)
            seen.add(k)

    lines: List[str] = []
    subj = subject.strip() or "This organism"

    for k in ordered:
        key = _title_case_field(str(k))
        raw = expected_fields.get(k)

        if isinstance(raw, list):
            vals = [x for x in _as_list(raw) if not _is_unknown(x)]
            if not vals:
                continue

            # Special handling for list-like fields
            if key == "Media Grown On":
                lines.append(f"{subj} can grow on: " + ", ".join(vals) + ".")
            elif key == "Colony Morphology":
                lines.append(f"{subj} colonies are described as: " + ", ".join(vals) + ".")
            else:
                lines.append(f"{subj} {key} includes: " + ", ".join(vals) + ".")
            continue

        val = _norm_str(raw)
        if _is_unknown(val):
            continue

        # Field-specific phrasing for better “evidence-like” feel
        if key == "Gram Stain":
            lines.append(f"{subj} is typically Gram {val}.")
        elif key == "Shape":
            lines.append(f"{subj} typically has shape: {val}.")
        elif key == "Oxygen Requirement":
            lines.append(f"{subj} is typically {val}.")
        elif key == "Growth Temperature":
            lines.append(f"{subj} typically grows within: {val} °C.")
        elif key == "Haemolysis Type":
            lines.append(f"{subj} haemolysis type is typically: {val}.")
        elif key == "Haemolysis":
            lines.append(f"{subj} haemolysis is typically: {val}.")
        elif key == "Pigment":
            if val.lower() in {"none", "no", "negative"}:
                lines.append(f"{subj} typically produces no pigment.")
            else:
                lines.append(f"{subj} may produce pigment: {val}.")
        elif key == "Colony Pattern":
            lines.append(f"{subj} colony/cellular pattern may be described as: {val}.")
        else:
            # Default: simple assertive sentence
            lines.append(f"{subj} {key} is typically: {val}.")

    # If we emitted nothing, return empty so we don’t add noise
    return "\n".join(lines).strip()

def _format_key_differentiators(items: List[Dict[str, Any]]) -> str:
    """
    For genus-level key_differentiators.
    """
    if not isinstance(items, list) or not items:
        return ""
    out: List[str] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        field = _norm_str(obj.get("field"))
        expected = _norm_str(obj.get("expected"))
        notes = _norm_str(obj.get("notes"))
        distinguishes_from = obj.get("distinguishes_from") or []
        if not field:
            continue

        line = f"{field}: expected {expected or 'Unknown'}."
        if isinstance(distinguishes_from, list) and distinguishes_from:
            line += " Distinguishes from: " + ", ".join([_norm_str(x) for x in distinguishes_from if _norm_str(x)])
            if not line.endswith("."):
                line += "."
        if notes:
            line += f" Notes: {notes}"
            if not line.endswith("."):
                line += "."
        out.append(line)

    return "\n".join(out)

def _format_common_confusions(items: List[Dict[str, Any]], level: str) -> str:
    """
    For genus/species common_confusions.
    """
    if not isinstance(items, list) or not items:
        return ""
    out: List[str] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        reason = _norm_str(obj.get("reason"))
        if level == "genus":
            who = _norm_str(obj.get("genus"))
            if who:
                out.append(f"{who}: {reason or 'Reason not specified.'}")
        else:
            who = _norm_str(obj.get("species")) or _norm_str(obj.get("genus"))
            if who:
                out.append(f"{who}: {reason or 'Reason not specified.'}")
    return "\n".join(out)

def _format_recommended_next_tests(items: List[Dict[str, Any]]) -> str:
    """
    For recommended_next_tests with optional API kit note.
    """
    if not isinstance(items, list) or not items:
        return ""
    out: List[str] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        test = _norm_str(obj.get("test"))
        reason = _norm_str(obj.get("reason"))
        api_kit = _norm_str(obj.get("api_kit"))

        if not test:
            continue

        line = f"{test}"
        if api_kit:
            line += f" (API kit: {api_kit})"
        if reason:
            line += f": {reason}"
        out.append(line)
    return "\n".join(out)

# ------------------------------------------------------------
# CHUNKING (SECTION-LOCAL)
# ------------------------------------------------------------

def chunk_text_by_paragraph(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """
    Chunk within a single section. We never merge different sections together.
    """
    text = (text or "").strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [l.strip() for l in text.splitlines() if l.strip()]

    chunks: List[str] = []
    current = ""

    for p in paras:
        candidate = (current + "\n\n" + p).strip() if current else p
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(p) <= max_chars:
                current = p
            else:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i + max_chars].strip())
                current = ""

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]

# ------------------------------------------------------------
# SECTION EMITTERS
# ------------------------------------------------------------

def emit_genus_sections(doc: Dict[str, Any], genus: str) -> List[Dict[str, Any]]:
    """
    Convert genus.json to a list of {section, role, text} entries.
    """
    out: List[Dict[str, Any]] = []

    overview = doc.get("overview") or {}
    if isinstance(overview, dict):
        short = _norm_str(overview.get("short"))
        clinical = _norm_str(overview.get("clinical_context"))
        if short:
            out.append({"section": "overview", "role": "description", "text": f"Genus {genus}: {short}"})
        if clinical:
            out.append({"section": "overview", "role": "description", "text": f"Clinical context: {clinical}"})

    expected_fields = doc.get("expected_fields")
    if isinstance(expected_fields, dict) and expected_fields:
        # 1) Declarative evidence-like sentences (NEW)
        sent = _expected_fields_to_sentences(expected_fields, subject=f"Genus {genus}")
        if sent:
            out.append({
                "section": "expected_profile_sentences",
                "role": "expected_profile",
                "text": sent,
            })

        # 2) Keep original key:value block (still useful)
        text = _format_expected_fields(expected_fields)
        if text:
            out.append({
                "section": "expected_fields",
                "role": "expected_profile",
                "text": f"Expected fields for genus {genus}:\n{text}",
            })

    field_notes = doc.get("field_notes")
    if isinstance(field_notes, dict) and field_notes:
        lines: List[str] = []
        for k in sorted(field_notes.keys(), key=lambda s: str(s).lower()):
            v = _norm_str(field_notes.get(k))
            if v:
                lines.append(f"{_title_case_field(str(k))}: {v}")
        if lines:
            out.append({"section": "field_notes", "role": "clarification", "text": "Field notes:\n" + "\n".join(lines)})

    kd = doc.get("key_differentiators")
    if isinstance(kd, list) and kd:
        text = _format_key_differentiators(kd)
        if text:
            out.append({"section": "key_differentiators", "role": "differentiation", "text": "Key differentiators:\n" + text})

    conf = doc.get("common_confusions")
    if isinstance(conf, list) and conf:
        text = _format_common_confusions(conf, level="genus")
        if text:
            out.append({"section": "common_confusions", "role": "warning", "text": "Common confusions:\n" + text})

    wq = doc.get("when_to_question_identification")
    if isinstance(wq, list) and wq:
        lines = [str(x).strip() for x in wq if str(x).strip()]
        if lines:
            out.append({"section": "when_to_question_identification", "role": "warning", "text": "When to question identification:\n" + _bullet_lines(lines)})

    rnt = doc.get("recommended_next_tests")
    if isinstance(rnt, list) and rnt:
        text = _format_recommended_next_tests(rnt)
        if text:
            out.append({"section": "recommended_next_tests", "role": "recommendation", "text": "Recommended next tests:\n" + text})

    ss = doc.get("supported_species")
    if isinstance(ss, list) and ss:
        species_list = [str(x).strip() for x in ss if str(x).strip()]
        if species_list:
            out.append({"section": "supported_species", "role": "metadata", "text": f"Supported species for genus {genus}: " + ", ".join(species_list)})

    return out


def emit_species_sections(doc: Dict[str, Any], genus: str, species: str) -> List[Dict[str, Any]]:
    """
    Convert a species JSON to a list of {section, role, text} entries.
    """
    out: List[Dict[str, Any]] = []
    overview = doc.get("overview") or {}
    if isinstance(overview, dict):
        short = _norm_str(overview.get("short"))
        clinical = _norm_str(overview.get("clinical_context"))
        if short:
            out.append({"section": "overview", "role": "description", "text": f"Species {genus} {species}: {short}"})
        if clinical:
            out.append({"section": "overview", "role": "description", "text": f"Clinical context: {clinical}"})

    expected_fields = doc.get("expected_fields")
    if isinstance(expected_fields, dict) and expected_fields:
        # 1) Declarative evidence-like sentences (NEW)
        sent = _expected_fields_to_sentences(expected_fields, subject=f"Species {genus} {species}")
        if sent:
            out.append({
                "section": "expected_profile_sentences",
                "role": "expected_profile",
                "text": sent,
            })

        # 2) Keep original key:value block
        text = _format_expected_fields(expected_fields)
        if text:
            out.append({"section": "expected_fields", "role": "expected_profile", "text": f"Expected fields for species {genus} {species}:\n{text}"})

    markers = doc.get("species_markers")
    if isinstance(markers, list) and markers:
        lines: List[str] = []
        for m in markers:
            if not isinstance(m, dict):
                continue
            field = _norm_str(m.get("field"))
            val = _norm_str(m.get("value"))
            importance = _norm_str(m.get("importance"))
            notes = _norm_str(m.get("notes"))
            if not field:
                continue
            line = f"{field}: {val or 'Unknown'}"
            if importance:
                line += f" (importance: {importance})"
            if notes:
                line += f" — {notes}"
            lines.append(line)
        if lines:
            out.append({"section": "species_markers", "role": "species_marker", "text": "Species markers:\n" + "\n".join(lines)})

    conf = doc.get("common_confusions")
    if isinstance(conf, list) and conf:
        text = _format_common_confusions(conf, level="species")
        if text:
            out.append({"section": "common_confusions", "role": "warning", "text": "Common confusions:\n" + text})

    wq = doc.get("when_to_question_identification")
    if isinstance(wq, list) and wq:
        lines = [str(x).strip() for x in wq if str(x).strip()]
        if lines:
            out.append({"section": "when_to_question_identification", "role": "warning", "text": "When to question identification:\n" + _bullet_lines(lines)})

    rnt = doc.get("recommended_next_tests")
    if isinstance(rnt, list) and rnt:
        text = _format_recommended_next_tests(rnt)
        if text:
            out.append({"section": "recommended_next_tests", "role": "recommendation", "text": "Recommended next tests:\n" + text})

    return out


# ------------------------------------------------------------
# INDEX BUILD
# ------------------------------------------------------------

def _iter_kb_files() -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    if not os.path.isdir(KB_ROOT):
        return entries

    for genus in sorted(os.listdir(KB_ROOT)):
        genus_dir = os.path.join(KB_ROOT, genus)
        if not os.path.isdir(genus_dir):
            continue
        for fname in sorted(os.listdir(genus_dir)):
            if fname.lower().endswith(".json"):
                entries.append((genus, os.path.join(genus_dir, fname)))
    return entries


def build_rag_index(max_chars: int = DEFAULT_MAX_CHARS) -> Dict[str, Any]:
    os.makedirs(INDEX_DIR, exist_ok=True)

    kb_entries = _iter_kb_files()
    if not kb_entries:
        return {"ok": False, "message": "No KB JSON files found."}

    docs_for_embedding: List[str] = []
    meta: List[Dict[str, Any]] = []

    num_json_errors = 0

    for genus_dir_name, path in kb_entries:
        with open(path, "r", encoding="utf-8") as f:
            try:
                doc = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[rag_index_builder] JSON error in {path}: {e}")
                num_json_errors += 1
                continue

        fname = os.path.basename(path)
        is_genus = fname == "genus.json"

        genus = _norm_str(doc.get("genus")) or genus_dir_name
        level = "genus" if is_genus else "species"

        species: Optional[str]
        if is_genus:
            species = None
            sections = emit_genus_sections(doc, genus=genus)
        else:
            species = _norm_str(doc.get("species")) or os.path.splitext(fname)[0]
            sections = emit_species_sections(doc, genus=genus, species=species)

        for sec in sections:
            section = _norm_str(sec.get("section"))
            role = _norm_str(sec.get("role"))
            text = _norm_str(sec.get("text"))

            if not section or not role or not text:
                continue

            chunks = chunk_text_by_paragraph(text, max_chars=max_chars)
            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                rec_id = f"{genus}|{species or 'GENUS'}|{section}|{idx}"

                docs_for_embedding.append(chunk)
                meta.append(
                    {
                        "id": rec_id,
                        "level": level,
                        "genus": genus,
                        "species": species,
                        "section": section,
                        "role": role,
                        "text": chunk,
                        "source_file": os.path.relpath(path),
                        "chunk_id": idx,
                        # Optional: helps later for field-level weighting
                        "field_key": None,
                    }
                )

    if not docs_for_embedding:
        return {
            "ok": False,
            "message": "No valid sections emitted from KB JSON files. Check schema/contents.",
            "num_files": len(kb_entries),
            "num_json_errors": num_json_errors,
        }

    embeddings = embed_texts(docs_for_embedding, normalize=True)

    index_records: List[Dict[str, Any]] = []
    for m, emb in zip(meta, embeddings):
        rec = dict(m)
        rec["embedding"] = emb.tolist()
        index_records.append(rec)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 2,
                "model_name": EMBEDDING_MODEL_NAME,
                "record_schema": {
                    "id": "str",
                    "level": "genus|species",
                    "genus": "str",
                    "species": "str|null",
                    "section": "str",
                    "role": "str",
                    "text": "str",
                    "source_file": "str",
                    "chunk_id": "int",
                    "embedding": "list[float]",
                },
                "stats": {
                    "num_files": len(kb_entries),
                    "num_records": len(index_records),
                    "num_json_errors": num_json_errors,
                    "chunk_max_chars": max_chars,
                },
                "records": index_records,
            },
            f,
            ensure_ascii=False,
        )

    return {
        "ok": True,
        "message": "RAG index built successfully (section-aware, declarative expected profiles).",
        "index_path": INDEX_PATH,
        "num_records": len(index_records),
        "num_files": len(kb_entries),
        "num_json_errors": num_json_errors,
        "chunk_max_chars": max_chars,
    }


if __name__ == "__main__":
    summary = build_rag_index()
    print(json.dumps(summary, indent=2))