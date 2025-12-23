# training/schema_expander.py
# ------------------------------------------------------------
# Stage 10C — SAFE schema expansion
#
# Core fields = EXACT columns in bacteria_db.xlsx.
# Extended fields = ONLY the ones NOT in DB and NOT in existing schema.
#
# This version:
#  - NEVER adds core fields to extended schema.
#  - Only adds true extended fields found in gold tests.
#  - Logs ambiguous or rare fields to proposals file.
#  - Reports field frequencies & values seen for debugging.
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
from typing import Dict, Any, List
from collections import Counter
from datetime import datetime

import pandas as pd

from engine.schema import (
    load_extended_schema,
    save_extended_schema,
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

GOLD_PATH = "training/gold_tests.json"
EXTENDED_SCHEMA_PATH = "data/extended_schema.json"
PROPOSALS_PATH = "data/extended_proposals.jsonl"

# Minimum frequency before auto-adding a new extended field
MIN_FIELD_FREQ = 5


# ------------------------------------------------------------
# Helper: load gold tests
# ------------------------------------------------------------

def _load_gold_tests() -> List[Dict[str, Any]]:
    if not os.path.exists(GOLD_PATH):
        return []
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []


# ------------------------------------------------------------
# Helper: load DB columns (TRUE core schema)
# ------------------------------------------------------------

def _load_db_columns() -> List[str]:
    candidates = [
        os.path.join("data", "bacteria_db.xlsx"),
        "bacteria_db.xlsx",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_excel(p)
                return [c.strip() for c in df.columns]
            except Exception:
                continue
    return []


# ------------------------------------------------------------
# Decide if field name is safe for auto-adding
# ------------------------------------------------------------

def _is_safe_field_name(name: str) -> bool:
    n = name.strip()
    if not n:
        return False

    low = n.lower()

    # Ignore extremely short or generic names
    if len(n) < 4:
        return False
    if low in {"test", "growth", "acid", "base", "value", "result"}:
        return False

    # Clear biochemical patterns
    patterns = [
        "hydrolysis",
        "fermentation",
        "decarboxylase",
        "dihydrolase",
        "reduction",
        "utilization",
        "tolerance",
        "solubility",
        "oxidation",
        "lysis",
        "susceptibility",
        "resistance",
        "pyruvate",
        "lecithinase",
        "lipase",
        "casein",
        "hippurate",
        "tyrosine",
    ]
    if any(pat in low for pat in patterns):
        return True

    # Known short disc tests
    known_short = {"CAMP", "PYR", "Optochin", "Bacitracin", "Novobiocin"}
    if n in known_short:
        return True

    # If contains "test" and more than one word → likely legitimate
    if "test" in low and " " in low:
        return True

    return False


# ------------------------------------------------------------
# Log proposal (rare/ambiguous fields)
# ------------------------------------------------------------

def _append_proposal(record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(PROPOSALS_PATH), exist_ok=True)
    with open(PROPOSALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# MAIN ENTRY — SAFE SCHEMA EXPANSION
# ------------------------------------------------------------

def expand_schema() -> Dict[str, Any]:
    gold = _load_gold_tests()
    if not gold:
        return {
            "ok": False,
            "message": f"No gold tests found at {GOLD_PATH}",
            "auto_added_fields": {},
            "proposed_fields": [],
            "schema_path": EXTENDED_SCHEMA_PATH,
            "proposals_path": PROPOSALS_PATH,
            "unknown_fields_raw": {},
            "field_frequencies": {},
        }

    db_columns = set(_load_db_columns())               # TRUE core schema
    extended_schema = load_extended_schema(EXTENDED_SCHEMA_PATH)
    extended_fields = set(extended_schema.keys())

    # Counter for unknown fields
    field_counts: Counter[str] = Counter()
    field_values: Dict[str, Counter[str]] = {}

    for test in gold:
        expected = test.get("expected", {})
        if not isinstance(expected, dict):
            continue

        for field, value in expected.items():
            fname = str(field).strip()
            if not fname:
                continue

            # Skip core DB fields
            if fname in db_columns:
                continue

            # Skip already-known extended fields
            if fname in extended_fields:
                continue

            # Count unknowns
            field_counts[fname] += 1
            if fname not in field_values:
                field_values[fname] = Counter()
            field_values[fname][str(value).strip()] += 1

    auto_added: Dict[str, Any] = {}
    proposed: List[Dict[str, Any]] = []

    # Decide which unknown fields to auto-add
    for fname, freq in field_counts.items():
        values_seen = dict(field_values.get(fname, {}))

        if freq >= MIN_FIELD_FREQ and _is_safe_field_name(fname):
            # Auto-add as extended test
            extended_schema[fname] = {
                "value_type": "enum_PNV",
                "description": "Auto-added from gold tests (Stage 10C)",
                "values": list(values_seen.keys()),
            }
            auto_added[fname] = {
                "freq": freq,
                "values_seen": list(values_seen.keys()),
            }
        else:
            # Log proposal for later review
            proposed.append(
                {
                    "field_name": fname,
                    "freq": freq,
                    "values_seen": values_seen,
                }
            )
            _append_proposal(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "field_name": fname,
                    "freq": freq,
                    "values_seen": values_seen,
                }
            )

    # Save updated schema
    if auto_added:
        save_extended_schema(extended_schema, EXTENDED_SCHEMA_PATH)

    return {
        "ok": True,
        "auto_added_fields": auto_added,
        "proposed_fields": proposed,
        "schema_path": EXTENDED_SCHEMA_PATH,
        "proposals_path": PROPOSALS_PATH,
        "unknown_fields_raw": {f: dict(cnt) for f, cnt in field_values.items()},
        "field_frequencies": dict(field_counts),
    }
