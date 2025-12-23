# training/gold_tester.py
# ------------------------------------------------------------
# Stage 10A: Evaluate parsers on gold tests.
# This MUST NOT crash during import.
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
from typing import Dict, Any, List

from engine.parser_rules import parse_text_rules
from engine.parser_ext import parse_text_extended


GOLD_PATH = "training/gold_tests.json"
REPORT_DIR = "reports"


def _load_gold_tests() -> List[Dict[str, Any]]:
    if not os.path.exists(GOLD_PATH):
        return []
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []


def run_gold_tests(mode: str = "rules") -> Dict[str, Any]:
    gold_tests = _load_gold_tests()
    if not gold_tests:
        return {
            "summary": {
                "mode": mode,
                "tests": 0,
                "total_correct": 0,
                "total_fields": 0,
                "overall_accuracy": 0.0,
                "proposals_path": "data/extended_proposals.jsonl",
            }
        }

    os.makedirs(REPORT_DIR, exist_ok=True)

    wrong_cases = []
    total_correct = 0
    total_fields = 0

    for idx, test in enumerate(gold_tests):
        text = test.get("input", "")
        expected = test.get("expected", {})

        if mode == "rules":
            parsed = parse_text_rules(text).get("parsed_fields", {})
        elif mode == "rules+extended":
            rule_fields = parse_text_rules(text).get("parsed_fields", {})
            ext_fields = parse_text_extended(text).get("parsed_fields", {})
            parsed = {**rule_fields, **ext_fields}
        else:
            parsed = {}

        # Compare field-by-field
        correct_count = 0
        for key, val in expected.items():
            total_fields += 1
            if key in parsed and str(parsed[key]).strip().lower() == str(val).strip().lower():
                correct_count += 1

        total_correct += correct_count

        if correct_count < len(expected):
            wrong_cases.append(idx)

    accuracy = total_correct / total_fields if total_fields else 0.0

    summary = {
        "mode": mode,
        "tests": len(gold_tests),
        "total_correct": total_correct,
        "total_fields": total_fields,
        "overall_accuracy": accuracy,
        "wrong_cases": wrong_cases,
        "proposals_path": "data/extended_proposals.jsonl",
    }

    return {"summary": summary}
