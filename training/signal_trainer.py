# training/signal_trainer.py
# ------------------------------------------------------------
# Stage 10C placeholder:
# Safely returns a no-op result for signal training.
# This MUST NOT crash during import.
# ------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Any
import json
import os


SIGNALS_PATH = "data/signals_catalog.json"


def train_signals() -> Dict[str, Any]:
    """
    Placeholder trainer. Does nothing except ensure signals_catalog.json exists.
    Must NEVER crash.
    """

    # Ensure signals catalog exists
    if not os.path.exists(SIGNALS_PATH):
        try:
            with open(SIGNALS_PATH, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return {
        "ok": True,
        "message": "Signal trainer not implemented yet (Stage 10C placeholder).",
        "signals_catalog_path": SIGNALS_PATH,
    }
