# training/gold_trainer.py
# ------------------------------------------------------------
# Stage 10C — Orchestrates gold-test-driven training:
#   1) Alias trainer (DISABLED for safety)
#   2) Schema expander (safe v10C)
#   3) Signals trainer     (placeholder)
#
# This file MUST successfully import and expose train_from_gold().
# ------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Any

# Safe schema expander
from training.schema_expander import expand_schema

# Placeholder signals trainer
from training.signal_trainer import train_signals


def train_from_gold() -> Dict[str, Any]:
    """
    Runs all gold-test–driven training components (Stage 10C).

    Returns a dict:
    {
      "alias_trainer": {...},
      "schema_expander": {...},
      "signals_trainer": {...}
    }
    """

    # --------------------------------------------------------
    # 1) Alias Trainer — DISABLED to avoid destructive mappings
    # --------------------------------------------------------
    alias_result = {
        "ok": False,
        "message": (
            "Alias trainer is disabled in Stage 10C to prevent unsafe "
            "auto-mappings. Edit data/alias_maps.json manually if needed."
        ),
        "alias_map_path": "data/alias_maps.json",
    }

    # --------------------------------------------------------
    # 2) Schema Expander — Safe version
    # --------------------------------------------------------
    try:
        schema_result = expand_schema()
    except Exception as e:
        schema_result = {
            "ok": False,
            "message": f"Schema expander crashed: {e}",
            "auto_added_fields": {},
            "proposed_fields": [],
            "schema_path": "data/extended_schema.json",
            "proposals_path": "data/extended_proposals.jsonl",
        }

    # --------------------------------------------------------
    # 3) Signals Trainer (placeholder)
    # --------------------------------------------------------
    try:
        signals_result = train_signals()
    except Exception as e:
        signals_result = {
            "ok": False,
            "message": f"Signal trainer crashed: {e}",
            "signals_catalog_path": "data/signals_catalog.json",
        }

    # --------------------------------------------------------
    # Combined report
    # --------------------------------------------------------
    return {
        "alias_trainer": alias_result,
        "schema_expander": schema_result,
        "signals_trainer": signals_result,
    }
