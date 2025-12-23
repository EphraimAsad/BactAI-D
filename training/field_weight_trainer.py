# training/field_weight_trainer.py
# ------------------------------------------------------------
# Stage 12A — Train Per-Field Parser Weights from Gold Tests
#
# Produces:
#   data/field_weights.json
#
# This script computes reliability scores for:
#   - parser_rules
#   - parser_ext
#   - parser_llm
#
# and outputs:
#   {
#     "global": { ... },
#     "fields": { field -> weights },
#     "meta": { ... }
#   }
#
# These weights are used by parser_fusion (Stage 12B).
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Core parsers
from engine.parser_rules import parse_text_rules
from engine.parser_ext import parse_text_extended

# LLM parser (optional)
try:
    from engine.parser_llm import parse_llm as parse_text_llm_local
except Exception:
    parse_text_llm_local = None  # gracefully degrade if LLM unavailable


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_GOLD_PATH = os.path.join("data", "gold_tests.json")
DEFAULT_OUT_PATH = os.path.join("data", "field_weights.json")

MISSING_PENALTY = 0.5
SMOOTHING = 1e-3


# ------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------

@dataclass
class ParserOutcome:
    prediction: Optional[str]
    correct: bool
    wrong: bool
    missing: bool


@dataclass
class FieldStats:
    correct: int = 0
    wrong: int = 0
    missing: int = 0

    def total(self) -> int:
        return self.correct + self.wrong + self.missing

    def score(self, missing_penalty: float = MISSING_PENALTY) -> float:
        if self.total() == 0:
            return 0.0
        denom = self.correct + self.wrong + missing_penalty * self.missing
        if denom == 0:
            return 0.0
        return self.correct / denom


# ------------------------------------------------------------
# Gold Loading
# ------------------------------------------------------------

def _load_gold_tests(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gold tests not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("gold_tests.json must be a list")
    return data


def _extract_text_and_expected(test_obj: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    text = (
        test_obj.get("text")
        or test_obj.get("description")
        or test_obj.get("input")
        or test_obj.get("raw")
        or ""
    )
    if not isinstance(text, str):
        text = str(text)

    expected: Dict[str, str] = {}

    if isinstance(test_obj.get("expected"), dict):
        for k, v in test_obj["expected"].items():
            expected[str(k)] = str(v)
        return text, expected

    if isinstance(test_obj.get("expected_core"), dict):
        for k, v in test_obj["expected_core"].items():
            expected[str(k)] = str(v)

    if isinstance(test_obj.get("expected_extended"), dict):
        for k, v in test_obj["expected_extended"].items():
            expected[str(k)] = str(v)

    return text, expected


# ------------------------------------------------------------
# Parser Execution
# ------------------------------------------------------------

def _get_parser_predictions(text: str, include_llm: bool = True) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}

    r = parse_text_rules(text)
    results["rules"] = dict(r.get("parsed_fields", {}))

    e = parse_text_extended(text)
    results["extended"] = dict(e.get("parsed_fields", {}))

    llm_values: Dict[str, str] = {}
    if include_llm and parse_text_llm_local is not None:
        try:
            llm_out = parse_text_llm_local(text)
            llm_values = dict(llm_out.get("parsed_fields", {}))
        except Exception:
            llm_values = {}
    results["llm"] = llm_values

    return results


def _outcome_for_field(expected_val: str, predicted_val: Optional[str]) -> ParserOutcome:
    if predicted_val is None:
        return ParserOutcome(prediction=None, correct=False, wrong=False, missing=True)
    if predicted_val == expected_val:
        return ParserOutcome(prediction=predicted_val, correct=True, wrong=False, missing=False)
    return ParserOutcome(prediction=predicted_val, correct=False, wrong=True, missing=False)


# ------------------------------------------------------------
# Stats Computation
# ------------------------------------------------------------

def _compute_stats_from_gold(
    gold_tests: List[Dict[str, Any]],
    include_llm: bool = True,
):
    field_stats = defaultdict(lambda: defaultdict(FieldStats))
    global_stats = defaultdict(FieldStats)

    total_samples = 0

    for sample in gold_tests:
        text, expected = _extract_text_and_expected(sample)
        if not expected:
            continue

        total_samples += 1
        preds = _get_parser_predictions(text, include_llm=include_llm)

        for field, expected_val in expected.items():
            expected_val = str(expected_val)
            for parser_name in ["rules", "extended", "llm"]:
                if parser_name == "llm" and not include_llm:
                    continue

                pred_val = preds.get(parser_name, {}).get(field)

                outcome = _outcome_for_field(expected_val, pred_val)

                fs = field_stats[field][parser_name]
                if outcome.correct:
                    fs.correct += 1
                if outcome.wrong:
                    fs.wrong += 1
                if outcome.missing:
                    fs.missing += 1

                gs = global_stats[parser_name]
                if outcome.correct:
                    gs.correct += 1
                if outcome.wrong:
                    gs.wrong += 1
                if outcome.missing:
                    gs.missing += 1

    return field_stats, global_stats, total_samples


def _normalise(weights: Dict[str, float], smoothing: float = SMOOTHING) -> Dict[str, float]:
    adjusted = {k: max(smoothing, v) for k, v in weights.items()}
    total = sum(adjusted.values())
    if total <= 0:
        n = len(adjusted)
        return {k: 1.0 / n for k in adjusted}
    return {k: v / total for k, v in adjusted.items()}


def _build_weights_json(
    field_stats,
    global_stats,
    total_samples,
    include_llm=True,
):
    # Global scores
    raw_global = {}
    for parser_name, stats in global_stats.items():
        if parser_name == "llm" and not include_llm:
            continue
        raw_global[parser_name] = stats.score(MISSING_PENALTY)

    global_weights = _normalise(raw_global)

    # Per-field
    fields_block = {}

    for field_name, stats_dict in field_stats.items():
        raw_scores = {}
        total_support = 0

        for parser_name, stats in stats_dict.items():
            if parser_name == "llm" and not include_llm:
                continue
            raw_scores[parser_name] = stats.score(MISSING_PENALTY)
            total_support += stats.total()

        if total_support < 5:
            # low support → blend global + local
            local_norm = _normalise(raw_scores)
            mixed = {}
            for p in global_weights:
                mixed[p] = 0.7 * global_weights[p] + 0.3 * local_norm.get(p, global_weights[p])
            field_w = _normalise(mixed)
        else:
            field_w = _normalise(raw_scores)

        fields_block[field_name] = {
            **field_w,
            "support": total_support,
        }

    return {
        "global": global_weights,
        "fields": fields_block,
        "meta": {
            "total_samples": total_samples,
            "missing_penalty": MISSING_PENALTY,
            "smoothing": SMOOTHING,
            "include_llm": include_llm,
        },
    }


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def train_field_weights(
    gold_path: str = DEFAULT_GOLD_PATH,
    out_path: str = DEFAULT_OUT_PATH,
    include_llm: bool = False,
):
    print(f"[12A] Loading gold tests: {gold_path}")
    gold = _load_gold_tests(gold_path)
    print(f"[12A] {len(gold)} gold samples loaded")

    field_stats, global_stats, total_samples = _compute_stats_from_gold(
        gold, include_llm=include_llm
    )

    print("[12A] Computing weights...")
    weights = _build_weights_json(
        field_stats, global_stats, total_samples, include_llm=include_llm
    )

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"[12A] Writing: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)

    print("[12A] Done.")
    return weights


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage 12A — Train parser weights")
    p.add_argument("--gold", type=str, default=DEFAULT_GOLD_PATH)
    p.add_argument("--out", type=str, default=DEFAULT_OUT_PATH)
    p.add_argument("--include-llm", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    train_field_weights(
        gold_path=args.gold,
        out_path=args.out,
        include_llm=args.include_llm,
    )


if __name__ == "__main__":
    main()
