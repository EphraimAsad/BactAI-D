# training/parser_eval.py
# ------------------------------------------------------------
# Parser Evaluation (Stage 10A)
#
# This version ONLY evaluates:
#   - Rule parser
#   - Extended parser
#
# The LLM parser is intentionally disabled at this stage
# because alias maps and schema are not trained yet.
#
# This makes Stage 10A FAST and stable (< 3 seconds).
# ------------------------------------------------------------

import json
import os
from typing import Dict, Any

from engine.parser_rules import parse_text_rules
from engine.parser_ext import parse_text_extended


# Path to the gold tests
GOLD_PATH = "training/gold_tests.json"


def evaluate_single_test(test: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate one gold test with rules + extended parsers.
    """
    text = test.get("input", "")
    expected = test.get("expected", {})

    # Run deterministic parsers
    rule_out = parse_text_rules(text).get("parsed_fields", {})
    ext_out = parse_text_extended(text).get("parsed_fields", {})

    # Merge rule + extended (extended overwrites rules)
    merged = dict(rule_out)
    for k, v in ext_out.items():
        if v != "Unknown":
            merged[k] = v

    total = len(expected)
    correct = 0
    wrong = {}

    for field, exp_val in expected.items():
        got = merged.get(field, "Unknown")
        if got.lower() == exp_val.lower():
            correct += 0 if exp_val == "Unknown" else 1   # Unknown is neutral
        else:
            wrong[field] = {"expected": exp_val, "got": got}

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0,
        "wrong": wrong,
        "merged": merged,
    }


def run_parser_eval(mode: str = "rules_extended") -> Dict[str, Any]:
    """
    Evaluate ALL gold tests using rules + extended parsing only.
    """
    if not os.path.exists(GOLD_PATH):
        return {"error": f"Gold test file not found at {GOLD_PATH}"}

    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold = json.load(f)

    results = []
    wrong_cases = []

    total_correct = 0
    total_fields = 0

    for test in gold:
        out = evaluate_single_test(test)
        results.append(out)

        total_correct += out["correct"]
        total_fields += out["total"]

        if out["wrong"]:
            wrong_cases.append({
                "name": test.get("name", "Unnamed"),
                "wrong": out["wrong"],
                "parsed": out["merged"],
                "expected": test.get("expected", {})
            })

    summary = {
        "mode": "rules+extended",
        "tests": len(gold),
        "total_correct": total_correct,
        "total_fields": total_fields,
        "overall_accuracy": total_correct / total_fields if total_fields else 0,
        "wrong_cases": wrong_cases,
    }

    return summary
