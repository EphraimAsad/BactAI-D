# rag/rag_generator.py
# ============================================================
# RAG generator using google/flan-t5-large (CPU-friendly)
#
# Goal (user-visible, structured, deterministic-first):
# - Show the user:
#     KEY TRAITS:
#     CONFLICTS:
#     CONCLUSION:
# - KEY TRAITS and CONFLICTS are extracted deterministically from the
#   shaped retriever context (preferred).
# - The LLM only writes the CONCLUSION (2–5 sentences) based on those
#   extracted sections.
#
# Reliability:
# - flan-t5 sometimes echoes prompt instructions.
# - We keep the prompt extremely short and avoid imperative bullet rules.
# - We keep deterministic fallback logic if the LLM output is garbage/echo.
#
# Expected usage:
#   ctx = retrieve_rag_context(..., parsed_fields=...)
#   explanation = generate_genus_rag_explanation(
#       phenotype_text=text,
#       rag_context=ctx.get("llm_context_shaped") or ctx.get("llm_context"),
#       genus=genus
#   )
#
# Optional HF Space logs:
#   export BACTAI_RAG_GEN_LOG_INPUT=1
#   export BACTAI_RAG_GEN_LOG_OUTPUT=1
# ============================================================

from __future__ import annotations

import os
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


# ------------------------------------------------------------
# MODEL CONFIG
# ------------------------------------------------------------

MODEL_NAME = "google/flan-t5-large"

_tokenizer: T5Tokenizer | None = None
_model: T5ForConditionalGeneration | None = None

# Keep small for CPU + to reduce prompt truncation weirdness
_MAX_INPUT_TOKENS = 768
_DEFAULT_MAX_NEW_TOKENS = 160

# Hard cap the context chars we feed to T5 (prevents the model focusing on junk)
_CONTEXT_CHAR_CAP = 2400


def _get_model() -> tuple[T5Tokenizer, T5ForConditionalGeneration]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        _model = T5ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float32,
        )
    return _tokenizer, _model


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
# PROMPT (LLM WRITES ONLY THE CONCLUSION)
# ------------------------------------------------------------

# Intentionally minimal. No "rules list", no bullets specification.
# The LLM sees ONLY extracted matches/conflicts and writes a short conclusion.
RAG_PROMPT = """summarize: Evaluate whether the phenotype fits the target genus using the provided matches and conflicts.

Target genus: {genus}

Key traits that match:
{matches}

Conflicts:
{conflicts}

Write a short conclusion (2–5 sentences) stating whether this is a strong, moderate, or tentative genus match, and briefly mention the most important matches and conflicts.
"""


# ------------------------------------------------------------
# OUTPUT CLEANUP + ECHO DETECTION
# ------------------------------------------------------------

_BAD_SUBSTRINGS = (
    "summarize:",
    "target genus",
    "key traits that match",
    "write a short conclusion",
    "conflicts:",
)

def _clean_generation(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    # collapse excessive whitespace/newlines
    s = re.sub(r"\s*\n+\s*", " ", s).strip()
    s = re.sub(r"\s{2,}", " ", s).strip()

    # guard runaway length
    if len(s) > 900:
        s = s[:900].rstrip() + "..."

    return s


def _looks_like_echo_or_garbage(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True

    # extremely short / non-sentence
    if len(s) < 25:
        return True

    low = s.lower()
    if any(bad in low for bad in _BAD_SUBSTRINGS):
        return True

    # Must look like actual prose
    if "." not in s and "because" not in low and "match" not in low and "fits" not in low:
        return True

    return False


# ------------------------------------------------------------
# EXTRACT KEY TRAITS + CONFLICTS FROM SHAPED CONTEXT
# ------------------------------------------------------------

# Shaped context format (example):
# KEY MATCHES:
# - Trait: Value (matches reference: ...)
#
# CONFLICTS (observed vs CORE traits):
# - Trait: Value (conflicts reference: ...)
# or:
# CONFLICTS: Not specified.

_KEY_MATCHES_HEADER_RE = re.compile(r"^\s*KEY MATCHES\s*:\s*$", re.IGNORECASE)
_CONFLICTS_HEADER_RE = re.compile(r"^\s*CONFLICTS\b.*:\s*$", re.IGNORECASE)
_CONFLICTS_INLINE_NONE_RE = re.compile(r"^\s*CONFLICTS\s*:\s*not specified\.?\s*$", re.IGNORECASE)

_MATCH_LINE_RE = re.compile(
    r"^\s*-\s*([^:]+)\s*:\s*(.+?)\s*\(matches reference:\s*(.+?)\)\s*$",
    re.IGNORECASE,
)
_CONFLICT_LINE_RE = re.compile(
    r"^\s*-\s*([^:]+)\s*:\s*(.+?)\s*\(conflicts reference:\s*(.+?)\)\s*$",
    re.IGNORECASE,
)

# More permissive bullet capture (if shaper changes slightly)
_GENERIC_BULLET_RE = re.compile(r"^\s*-\s*(.+?)\s*$")


def _extract_key_traits_and_conflicts(shaped_ctx: str) -> tuple[list[str], list[str], bool]:
    """
    Extracts KEY MATCHES and CONFLICTS bullets from shaped retriever context.

    Returns:
      (key_traits, conflicts, found_structured_headers)

    - key_traits items are short: "Trait: ObservedValue"
    - conflicts items are short: "Trait: ObservedValue"
    """
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

        # detect headers
        if _KEY_MATCHES_HEADER_RE.match(line.strip()):
            in_matches = True
            in_conflicts = False
            saw_headers = True
            continue

        if _CONFLICTS_INLINE_NONE_RE.match(line.strip()):
            in_matches = False
            in_conflicts = False
            saw_headers = True
            # explicit "no conflicts"
            continue

        if _CONFLICTS_HEADER_RE.match(line.strip()):
            in_matches = False
            in_conflicts = True
            saw_headers = True
            continue

        # stop capture if another section begins (common shaper headings)
        if saw_headers and (line.strip().endswith(":") and not line.strip().startswith("-")):
            # If it's a new heading (and not one of our two), stop both
            if not _KEY_MATCHES_HEADER_RE.match(line.strip()) and not _CONFLICTS_HEADER_RE.match(line.strip()):
                in_matches = False
                in_conflicts = False

        # capture bullets under each section
        if in_matches and line.strip().startswith("-"):
            m = _MATCH_LINE_RE.match(line.strip())
            if m:
                trait = m.group(1).strip()
                obs = m.group(2).strip()
                key_traits.append(f"{trait}: {obs}")
            else:
                g = _GENERIC_BULLET_RE.match(line.strip())
                if g:
                    key_traits.append(g.group(1).strip())
            continue

        if in_conflicts and line.strip().startswith("-"):
            c = _CONFLICT_LINE_RE.match(line.strip())
            if c:
                trait = c.group(1).strip()
                obs = c.group(2).strip()
                conflicts.append(f"{trait}: {obs}")
            else:
                g = _GENERIC_BULLET_RE.match(line.strip())
                if g:
                    conflicts.append(g.group(1).strip())
            continue

    return key_traits, conflicts, saw_headers


def _extract_matches_conflicts_legacy(shaped_ctx: str) -> tuple[list[str], list[str]]:
    """
    Legacy extraction based purely on (matches reference: ...) / (conflicts reference: ...)
    anywhere in the text. Useful if headers are missing.
    """
    matches: list[str] = []
    conflicts: list[str] = []

    for raw in (shaped_ctx or "").splitlines():
        line = raw.strip()
        if not line.startswith("-"):
            continue

        m = _MATCH_LINE_RE.match(line)
        if m:
            trait = m.group(1).strip()
            obs = m.group(2).strip()
            matches.append(f"{trait}: {obs}")
            continue

        c = _CONFLICT_LINE_RE.match(line)
        if c:
            trait = c.group(1).strip()
            obs = c.group(2).strip()
            conflicts.append(f"{trait}: {obs}")
            continue

    return matches, conflicts


def _format_bullets(items: list[str], *, none_text: str) -> str:
    if not items:
        return none_text
    return "\n".join(f"- {x}" for x in items)


# ------------------------------------------------------------
# DETERMINISTIC CONCLUSION FALLBACK
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
            f"{', '.join(m)}. No major conflicts were detected against the retrieved core genus traits, "
            f"supporting a strong genus-level match."
        )
    if (not m) and c:
        return (
            f"This phenotype does not cleanly fit {g} because it conflicts with core traits "
            f"({', '.join(c)}). Consider re-checking those tests or comparing against the next-ranked genera."
        )

    return (
        f"Reference evidence was available for {g}, but no clear matches or conflicts could be extracted "
        f"from the shaped context. Try increasing top_k genus chunks or ensuring parsed_fields are being "
        f"passed into retrieve_rag_context so the shaper can compute KEY MATCHES and CONFLICTS."
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
    """
    Generates a structured RAG output intended for direct display:

      KEY TRAITS:
      - ...
      CONFLICTS:
      - ...
      CONCLUSION:
      ...

    Notes:
    - KEY TRAITS + CONFLICTS are extracted deterministically from the (shaped) context.
    - The LLM writes only the CONCLUSION.
    - If the LLM output is garbage/echo, we use a deterministic conclusion fallback.
    """
    tokenizer, model = _get_model()

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

    # Prefer structured extraction (KEY MATCHES / CONFLICTS sections)
    key_traits, conflicts, saw_headers = _extract_key_traits_and_conflicts(context)

    # If the headers weren't found or extraction is empty, try legacy extraction
    if (not saw_headers) or (not key_traits and not conflicts):
        legacy_matches, legacy_conflicts = _extract_matches_conflicts_legacy(context)
        if legacy_matches or legacy_conflicts:
            key_traits = key_traits or legacy_matches
            conflicts = conflicts or legacy_conflicts

    key_traits_text = _format_bullets(key_traits, none_text="- Not specified.")
    conflicts_text = _format_bullets(conflicts, none_text="- Not specified.")

    # LLM: conclusion only
    prompt = RAG_PROMPT.format(
        genus=genus_clean,
        matches=key_traits_text,
        conflicts=conflicts_text,
    )

    if RAG_GEN_LOG_INPUT:
        _log_block("PROMPT (CONCLUSION-ONLY)", prompt[:3000] + ("\n...(truncated)" if len(prompt) > 3000 else ""))

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_INPUT_TOKENS,
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        num_beams=1,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    cleaned = _clean_generation(decoded)

    if RAG_GEN_LOG_OUTPUT:
        _log_block("RAW OUTPUT (CONCLUSION)", decoded)
        _log_block("CLEANED OUTPUT (CONCLUSION)", cleaned)

    # If LLM output is junk, use deterministic conclusion
    if _looks_like_echo_or_garbage(cleaned):
        cleaned = _deterministic_conclusion(genus_clean, key_traits, conflicts)
        if RAG_GEN_LOG_OUTPUT:
            _log_block("FALLBACK CONCLUSION (DETERMINISTIC)", cleaned)

    # Final user-visible structured output
    final = (
        "KEY TRAITS:\n"
        f"{key_traits_text}\n\n"
        "CONFLICTS:\n"
        f"{conflicts_text}\n\n"
        "CONCLUSION:\n"
        f"{cleaned}"
    )

    return final