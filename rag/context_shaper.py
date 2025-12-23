# rag/context_shaper.py
# ============================================================
# Context shaper for RAG
#
# Goal:
# - Convert "flattened schema dumps" (Field: Value lines) into
#   readable evidence blocks the LLM can reason over.
# - Deterministic, no LLM usage.
#
# Works with:
# - llm_context from rag_retriever (biology-only text)
# ============================================================

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional


_FIELD_LINE_RE = re.compile(r"^\s*([^:\n]{1,80})\s*:\s*(.+?)\s*$")

# Some fields are usually lists separated by ; or , or |
_LIST_LIKE_FIELDS = {
    "Media Grown On",
    "Colony Morphology",
    "Colony Pattern",
    "Growth Temperature",
}

# A light grouping map to turn fields into readable sections.
# (You can expand this over time.)
_GROUPS: List[Tuple[str, List[str]]] = [
    ("Morphology & staining", [
        "Gram Stain", "Shape", "Cellular Arrangement", "Capsule", "Spore Forming",
    ]),
    ("Culture & colony", [
        "Media Grown On", "Colony Morphology", "Colony Pattern", "Pigment", "Odour",
        "Haemolysis", "Haemolysis Type",
    ]),
    ("Core biochemistry", [
        "Oxidase", "Catalase", "Indole", "Urease", "Citrate", "Methyl Red", "VP",
        "Nitrate Reduction", "ONPG", "TSI Pattern", "H2S", "Gas Production",
        "Glucose Fermentation", "Lactose Fermentation", "Sucrose Fermentation",
        "Inositol Fermentation", "Mannitol Fermentation",
    ]),
    ("Motility & growth conditions", [
        "Motility", "Motility Type", "Growth Temperature", "NaCl", "NaCl Tolerance",
        "Oxygen Requirement",
    ]),
    ("Other tests", [
        "DNase", "Esculin Hydrolysis", "Gelatin Hydrolysis",
        "Lysine Decarboxylase", "Ornithine Decarboxylase", "Arginine Dihydrolase",
    ]),
]


def _is_schema_dump(text: str) -> bool:
    """
    Detect if rag context looks like flattened Field: Value lines.
    """
    if not text:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 6:
        return False
    hits = 0
    for l in lines[:40]:
        if _FIELD_LINE_RE.match(l):
            hits += 1
    return hits >= max(4, int(0.5 * min(len(lines), 40)))


def _split_listish(field: str, value: str) -> str:
    """
    Normalize list-like values into comma-separated readable text.
    """
    v = (value or "").strip()
    if not v:
        return v
    if field in _LIST_LIKE_FIELDS or (";" in v) or ("," in v):
        parts = [p.strip() for p in re.split(r"[;,\|]+", v) if p.strip()]
        if parts:
            return ", ".join(parts)
    return v


def _parse_field_lines(text: str) -> Dict[str, str]:
    """
    Parse Field: Value lines into a dict. Keeps last occurrence.
    """
    out: Dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _FIELD_LINE_RE.match(line)
        if not m:
            continue
        field = m.group(1).strip()
        value = m.group(2).strip()
        if not field:
            continue
        out[field] = _split_listish(field, value)
    return out


def _format_grouped_blocks(fields: Dict[str, str]) -> str:
    """
    Turn fields into grouped, readable evidence blocks.
    """
    used = set()
    blocks: List[str] = []

    for title, keys in _GROUPS:
        lines: List[str] = []
        for k in keys:
            if k in fields:
                val = fields[k]
                if val and val.lower() != "unknown":
                    lines.append(f"- {k}: {val}")
                used.add(k)
        if lines:
            blocks.append(f"{title}:\n" + "\n".join(lines))

    # Any leftovers not in group map
    leftovers: List[str] = []
    for k, v in fields.items():
        if k in used:
            continue
        if not v or v.lower() == "unknown":
            continue
        leftovers.append(f"- {k}: {v}")
    if leftovers:
        blocks.append("Additional traits:\n" + "\n".join(leftovers))

    return "\n\n".join(blocks).strip()


def shape_llm_context(
    llm_context: str,
    target_genus: str = "",
    max_chars: int = 1800,
) -> str:
    """
    Main entrypoint.
    - If context is already narrative, keep it (trim to max_chars).
    - If it is a schema dump, convert to grouped evidence blocks.
    """
    ctx = (llm_context or "").strip()
    if not ctx:
        return ""

    if _is_schema_dump(ctx):
        fields = _parse_field_lines(ctx)
        shaped = _format_grouped_blocks(fields)

        # Add a tiny header to cue the LLM that this is reference evidence
        if target_genus:
            shaped = f"Reference evidence for {target_genus} (compiled traits):\n\n{shaped}"
        else:
            shaped = f"Reference evidence (compiled traits):\n\n{shaped}"

        return shaped[:max_chars].strip()

    # Narrative context: just trim
    if target_genus:
        ctx = f"Reference context for {target_genus}:\n\n{ctx}"
    return ctx[:max_chars].strip()