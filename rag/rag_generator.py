# rag/rag_generator.py
# ============================================================
# RAG generator using Ollama (Llama 3.2 3B by default)
#
# Evolution (confidence-aware, deterministic-first):
#
# - Confidence and escalation decisions are now made upstream in the
#   retriever context shaper (CONFIDENCE ASSESSMENT block).
#
# - The generator now:
#     1) Extracts:
#           KEY TRAITS
#           CONFLICTS
#           CONFIDENCE STATE
#           RECOMMENDED ACTION
#     2) Selects a deterministic conclusion template
#     3) Optionally allows the LLM to paraphrase the conclusion
#        while preserving the confidence state meaning
#
# - The LLM is *not allowed* to decide strength — it only
#   rewrites the conclusion within the declared state.
#
# - If the LLM returns junk → deterministic template is used.
#
# ============================================================

from __future__ import annotations

import os
import re

from rag.ollama_client import (
    generate as ollama_generate,
    is_ollama_running,
    health_check as ollama_health_check,
    OLLAMA_AVAILABLE,
    DEFAULT_MODEL as OLLAMA_MODEL,
)


# ------------------------------------------------------------
# MODEL CONFIG
# ------------------------------------------------------------

# Ollama model name (can be overridden via environment variable)
MODEL_NAME = OLLAMA_MODEL

_DEFAULT_MAX_NEW_TOKENS = 256

_CONTEXT_CHAR_CAP = 2400


def _check_ollama() -> bool:
    """Check if Ollama is available and running."""
    if not OLLAMA_AVAILABLE:
        return False
    return is_ollama_running()


# ------------------------------------------------------------
# DEBUG LOGGING (HF Space logs)
# ------------------------------------------------------------

RAG_GEN_LOG_INPUT = os.getenv("BACTAI_RAG_GEN_LOG_INPUT", "0").strip() == "1"
RAG_GEN_LOG_OUTPUT = os.getenv("BACTAI_RAG_GEN_LOG_OUTPUT", "0").strip() == "1"


def _log_block(title: str, body: str) -> None:
    print("=" * 80)
    print(f"RAG GENERATOR DEBUG — {title}")
    print("=" * 80)
    print(body.strip() if body else "")
    print()


# ------------------------------------------------------------
# PROMPT (LLM MAY PARAPHRASE — BUT NOT CHANGE CONFIDENCE)
# ------------------------------------------------------------

RAG_PROMPT = """You are a microbiology assistant. Rewrite the following conclusion about bacterial identification.

RULES:
- Do NOT change the confidence level
- Do NOT introduce new reasoning or facts
- Write exactly 2-4 short sentences in clear, natural language
- Mention both matches and conflicts when present

TARGET GENUS: {genus}
CONFIDENCE STATE: {confidence_state}
RECOMMENDED ACTION: {recommended_action}

KEY TRAITS THAT MATCH:
{matches}

CONFLICTS:
{conflicts}

Write a concise conclusion that preserves the confidence state meaning. Only output the conclusion, nothing else."""


# ------------------------------------------------------------
# OUTPUT CLEANUP / GARBAGE DETECTION
# ------------------------------------------------------------

_BAD_SUBSTRINGS = (
    "summarize:",
    "target genus",
    "confidence state",
    "write a concise conclusion",
    "conflicts:",
)

def _clean_generation(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    s = re.sub(r"\s*\n+\s*", " ", s).strip()
    s = re.sub(r"\s{2,}", " ", s).strip()

    if len(s) > 900:
        s = s[:900].rstrip() + "..."

    return s


def _looks_like_echo_or_garbage(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    if len(s) < 25:
        return True

    low = s.lower()
    if any(bad in low for bad in _BAD_SUBSTRINGS):
        return True

    if "." not in s and "match" not in low and "conflict" not in low:
            return True

    return False


# ------------------------------------------------------------
# EXTRACT KEY TRAITS + CONFLICTS
# ------------------------------------------------------------

_KEY_MATCHES_HEADER_RE = re.compile(r"^\s*KEY MATCHES\s*:\s*$", re.IGNORECASE)
_CONFLICTS_HEADER_RE = re.compile(r"^\s*CONFLICTS\b.*:\s*$", re.IGNORECASE)
_CONFLICTS_INLINE_NONE_RE = re.compile(
    r"^\s*CONFLICTS\s*:\s*not specified\.?\s*$",
    re.IGNORECASE,
)

_MATCH_LINE_RE = re.compile(
    r"^\s*-\s*([^:]+)\s*:\s*(.+?)\s*\(matches reference:\s*(.+?)\)\s*$",
    re.IGNORECASE,
)

_CONFLICT_LINE_RE = re.compile(
    r"^\s*-\s*([^:]+)\s*:\s*(.+?)\s*\(conflicts reference:\s*(.+?)\)\s*$",
    re.IGNORECASE,
)

_GENERIC_BULLET_RE = re.compile(r"^\s*-\s*(.+?)\s*$")


def _extract_key_traits_and_conflicts(
    shaped_ctx: str,
) -> tuple[list[str], list[str], bool]:

    key_traits: list[str] = []
    conflicts: list[str] = []

    lines = (shaped_ctx or "").splitlines()
    if not lines:
        return key_traits, conflicts, False

    in_matches = False
    in_conflicts = False
    saw_headers = False

    for raw in lines:
        line = raw.rstrip("\n")

        if _KEY_MATCHES_HEADER_RE.match(line.strip()):
            in_matches, in_conflicts = True, False
            saw_headers = True
            continue

        if _CONFLICTS_INLINE_NONE_RE.match(line.strip()):
            in_matches = in_conflicts = False
            saw_headers = True
            continue

        if _CONFLICTS_HEADER_RE.match(line.strip()):
            in_matches, in_conflicts = False, True
            saw_headers = True
            continue

        if saw_headers and (line.strip().endswith(":") and not line.strip().startswith("-")):
            if not _KEY_MATCHES_HEADER_RE.match(line.strip()) and not _CONFLICTS_HEADER_RE.match(line.strip()):
                in_matches = in_conflicts = False

        if in_matches and line.strip().startswith("-"):
            m = _MATCH_LINE_RE.match(line.strip())
            if m:
                key_traits.append(f"{m.group(1).strip()}: {m.group(2).strip()}")
            else:
                g = _GENERIC_BULLET_RE.match(line.strip())
                if g:
                    key_traits.append(g.group(1).strip())
            continue

        if in_conflicts and line.strip().startswith("-"):
            c = _CONFLICT_LINE_RE.match(line.strip())
            if c:
                conflicts.append(f"{c.group(1).strip()}: {c.group(2).strip()}")
            else:
                g = _GENERIC_BULLET_RE.match(line.strip())
                if g:
                    conflicts.append(g.group(1).strip())
            continue

    return key_traits, conflicts, saw_headers


def _extract_matches_conflicts_legacy(shaped_ctx: str) -> tuple[list[str], list[str]]:
    matches: list[str] = []
    conflicts: list[str] = []

    for raw in (shaped_ctx or "").splitlines():
        line = raw.strip()
        if not line.startswith("-"):
            continue

        m = _MATCH_LINE_RE.match(line)
        if m:
            matches.append(f"{m.group(1).strip()}: {m.group(2).strip()}")
            continue

        c = _CONFLICT_LINE_RE.match(line)
        if c:
            conflicts.append(f"{c.group(1).strip()}: {c.group(2).strip()}")
            continue

    return matches, conflicts


def _format_bullets(items: list[str], *, none_text: str) -> str:
    if not items:
        return none_text
    return "\n".join(f"- {x}" for x in items)


# ------------------------------------------------------------
# CONFIDENCE STATE EXTRACTOR
# ------------------------------------------------------------

_CONF_STATE_RE = re.compile(r"^\s*-\s*Confidence State:\s*(.+?)\s*$", re.IGNORECASE)
_RECOMMENDED_ACTION_RE = re.compile(r"^\s*-\s*Recommended Action:\s*(.+?)\s*$", re.IGNORECASE)

def _extract_confidence_state(shaped_ctx: str) -> tuple[str | None, str | None]:
    confidence_state = None
    recommended_action = None

    for raw in (shaped_ctx or "").splitlines():
        line = raw.strip()

        m = _CONF_STATE_RE.match(line)
        if m:
            confidence_state = m.group(1).strip()
            continue

        r = _RECOMMENDED_ACTION_RE.match(line)
        if r:
            recommended_action = r.group(1).strip()
            continue

    return confidence_state, recommended_action


# ------------------------------------------------------------
# DETERMINISTIC TEMPLATES
# ------------------------------------------------------------

def _template_conclusion(
    genus: str,
    confidence_state: str | None,
    key_traits: list[str],
    conflicts: list[str],
    recommended_action: str | None,
) -> str:

    g = (genus or "").strip() or "Unknown"
    rec = recommended_action or ""

    m_short = ", ".join(key_traits[:3]) if key_traits else "no clearly supportive traits"
    c_short = ", ".join(conflicts[:3]) if conflicts else None

    if confidence_state is None:
        return _deterministic_conclusion(g, key_traits, conflicts)

    cs = confidence_state.lower()

    if "strong" in cs:
        return (
            f"This phenotype is indicative of {g} with no conflicting traits observed. "
            f"It aligns well with key genus characteristics such as {m_short}. "
            f"{rec}".strip()
        )

    if "probable" in cs or "conflicts present" in cs or "cautious" in cs:
        if c_short:
            return (
                f"The phenotype is consistent with {g} based on supporting traits including {m_short}; "
                f"however, conflicting results are present ({c_short}), which reduces confidence. "
                f"{rec}".strip()
            )
        else:
            return (
                f"The phenotype is broadly consistent with {g}, but limited conflicting information "
                f"reduces certainty. {rec}".strip()
            )

    if "inconclusive" in cs or "conflicting" in cs:
        return (
            f"The top genus match is {g}; however, the phenotype is inconclusive due to conflicting "
            f"test results ({c_short or 'multiple conflicting traits'}). {rec}".strip()
        )

    if "weak" in cs:
        return (
            f"The available phenotype provides weak evidence for {g}. "
            f"Additional testing or phenotype data is recommended. {rec}".strip()
        )

    return _deterministic_conclusion(g, key_traits, conflicts)


# ------------------------------------------------------------
# BACKSTOP DETERMINISTIC CONCLUSION
# ------------------------------------------------------------

def _deterministic_conclusion(genus: str, key_traits: list[str], conflicts: list[str]) -> str:
    g = (genus or "").strip() or "Unknown"
    m = key_traits[:4]
    c = conflicts[:2]

    if m and c:
        return (
            f"This is a probable match to {g} because it aligns with key traits such as "
            f"{', '.join(m)}. However, there are conflicts ({', '.join(c)}), so treat this "
            f"as a moderate/tentative genus-level fit and consider re-checking the conflicting tests."
        )
    if m and not c:
        return (
            f"This phenotype is consistent with {g} based on key matching traits such as "
            f"{', '.join(m)}. No major conflicts were detected against the core genus traits, "
            f"supporting a strong genus-level match."
        )
    if (not m) and c:
        return (
            f"This phenotype does not cleanly fit {g} because it conflicts with core traits "
            f"({', '.join(c)}). Consider re-checking those tests or comparing against the next-ranked genera."
        )

    return (
        f"Reference evidence was available for {g}, but no clear matches or conflicts could be extracted "
        f"from the shaped context."
    )


def _trim_context(ctx: str) -> str:
    s = (ctx or "").strip()
    if not s:
        return ""
    if len(s) <= _CONTEXT_CHAR_CAP:
        return s
    return s[:_CONTEXT_CHAR_CAP].rstrip() + "\n... (truncated)"


# ------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------

def generate_genus_rag_explanation(
    phenotype_text: str,
    rag_context: str,
    genus: str,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
) -> str:

    genus_clean = (genus or "").strip() or "Unknown"
    context = _trim_context(rag_context or "")

    if not context:
        return (
            "KEY TRAITS:\n"
            "- Not specified.\n\n"
            "CONFLICTS:\n"
            "- Not specified.\n\n"
            "CONCLUSION:\n"
            "No reference evidence was available to evaluate this genus against the observed phenotype."
        )

    key_traits, conflicts, saw_headers = _extract_key_traits_and_conflicts(context)

    if (not saw_headers) or (not key_traits and not conflicts):
        legacy_matches, legacy_conflicts = _extract_matches_conflicts_legacy(context)
        if legacy_matches or legacy_conflicts:
            key_traits = key_traits or legacy_matches
            conflicts = conflicts or legacy_conflicts

    key_traits_text = _format_bullets(key_traits, none_text="- Not specified.")
    conflicts_text = _format_bullets(conflicts, none_text="- Not specified.")

    confidence_state, recommended_action = _extract_confidence_state(context)

    template_conclusion = _template_conclusion(
        genus_clean,
        confidence_state,
        key_traits,
        conflicts,
        recommended_action,
    )

    if confidence_state is None:
        final_conclusion = template_conclusion

    else:
        prompt = RAG_PROMPT.format(
            genus=genus_clean,
            confidence_state=confidence_state,
            recommended_action=recommended_action or "None",
            matches=key_traits_text,
            conflicts=conflicts_text,
        )

        if RAG_GEN_LOG_INPUT:
            _log_block("PROMPT (CONFIDENCE-AWARE)", prompt)

        # Check if Ollama is available; if not, use template
        if not _check_ollama():
            final_conclusion = template_conclusion
            if RAG_GEN_LOG_OUTPUT:
                _log_block("FALLBACK (OLLAMA NOT AVAILABLE)", final_conclusion)
        else:
            try:
                # Generate using Ollama
                decoded = ollama_generate(
                    prompt=prompt,
                    model=MODEL_NAME,
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                )
                cleaned = _clean_generation(decoded)

                if RAG_GEN_LOG_OUTPUT:
                    _log_block("RAW OUTPUT (CONCLUSION)", decoded)
                    _log_block("CLEANED OUTPUT", cleaned)

                if _looks_like_echo_or_garbage(cleaned):
                    final_conclusion = template_conclusion
                    if RAG_GEN_LOG_OUTPUT:
                        _log_block("FALLBACK (TEMPLATE)", final_conclusion)
                else:
                    final_conclusion = cleaned

            except Exception as e:
                # Fallback to template on any Ollama error
                final_conclusion = template_conclusion
                if RAG_GEN_LOG_OUTPUT:
                    _log_block("FALLBACK (OLLAMA ERROR)", f"{e}\n\n{final_conclusion}")

    final = (
        "KEY TRAITS:\n"
        f"{key_traits_text}\n\n"
        "CONFLICTS:\n"
        f"{conflicts_text}\n\n"
        "CONCLUSION:\n"
        f"{final_conclusion}"
    )

    return final


# ------------------------------------------------------------
# OLLAMA STATUS (for backend health checks)
# ------------------------------------------------------------

def get_ollama_status() -> dict:
    """Get Ollama health status for API endpoints."""
    return ollama_health_check()