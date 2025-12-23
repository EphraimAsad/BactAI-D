# engine/bacteria_identifier.py
# ------------------------------------------------------------
# Core BactAI-D identification engine.
# - Scores genera from Excel DB (core phenotype fields)
# - Integrates optional extended-test reasoning
# - Provides blended confidence and narrative reasoning
# ------------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

try:
    from engine.extended_reasoner import score_genera_from_extended
    HAS_EXTENDED_REASONER = True
except Exception:
    HAS_EXTENDED_REASONER = False


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def join_with_and(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# ------------------------------------------------------------
# Identification Result
# ------------------------------------------------------------

@dataclass
class IdentificationResult:
    genus: str
    total_score: int
    matched_fields: List[str] = field(default_factory=list)
    mismatched_fields: List[str] = field(default_factory=list)
    reasoning_factors: Dict[str, Any] = field(default_factory=dict)
    total_fields_evaluated: int = 0
    total_fields_possible: int = 0
    extra_notes: str = ""
    extended_score: float = 0.0  # 0.0–1.0
    extended_explanation: str = ""

    # ---------- Confidence metrics ----------

    def confidence_percent(self) -> int:
        """Confidence based only on tests the user entered."""
        if self.total_fields_evaluated <= 0:
            return 0
        pct = (self.total_score / max(1, self.total_fields_evaluated)) * 100
        return max(0, min(100, int(round(pct))))

    def true_confidence(self) -> int:
        """Confidence based on all possible fields in the DB."""
        if self.total_fields_possible <= 0:
            return 0
        pct = (self.total_score / max(1, self.total_fields_possible)) * 100
        return max(0, min(100, int(round(pct))))

    def blended_confidence_percent(self) -> int:
        """
        Blend core confidence with extended_score (0–1).
        If no extended signal, return core confidence.
        Simple blend: 70% core, 30% extended signal.
        """
        core = self.confidence_percent()
        if self.extended_score <= 0:
            return core

        ext_pct = max(0.0, min(1.0, float(self.extended_score))) * 100.0
        blended = 0.7 * core + 0.3 * ext_pct
        return max(0, min(100, int(round(blended))))

    # ---------- Reasoning text ----------

    def reasoning_paragraph(self, ranked_results: Optional[List["IdentificationResult"]] = None) -> str:
        """Generate a narrative explanation from core matches."""
        if not self.matched_fields and not self.reasoning_factors:
            return "No significant biochemical or morphological matches were found."

        intro_options = [
            "Based on the observed biochemical and morphological traits,",
            "According to the provided test results,",
            "From the available laboratory findings,",
            "Considering the entered reactions and colony characteristics,",
        ]

        import random
        intro = random.choice(intro_options)

        highlights = []

        gram = self.reasoning_factors.get("Gram Stain")
        if gram:
            highlights.append(f"it is **Gram {str(gram).lower()}**")

        shape = self.reasoning_factors.get("Shape")
        if shape:
            highlights.append(f"with a **{str(shape).lower()}** morphology")

        catalase = self.reasoning_factors.get("Catalase")
        if catalase:
            highlights.append(f"and **catalase {str(catalase).lower()}** activity")

        oxidase = self.reasoning_factors.get("Oxidase")
        if oxidase:
            highlights.append(f"and **oxidase {str(oxidase).lower()}** reaction")

        oxy = self.reasoning_factors.get("Oxygen Requirement")
        if oxy:
            highlights.append(f"which prefers **{str(oxy).lower()}** conditions")

        if len(highlights) > 1:
            summary = ", ".join(highlights[:-1]) + " and " + highlights[-1]
        else:
            summary = "".join(highlights)

        # Confidence text (core)
        core_conf = self.confidence_percent()
        if core_conf >= 70:
            confidence_text = "The confidence in this identification is high."
        elif core_conf >= 40:
            confidence_text = "The confidence in this identification is moderate."
        else:
            confidence_text = "The confidence in this identification is low."

        # Comparison vs other top results
        comparison = ""
        if ranked_results and len(ranked_results) > 1:
            close_others = ranked_results[1:3]
            other_names = [r.genus for r in close_others]
            if other_names:
                if self.total_score >= close_others[0].total_score:
                    comparison = (
                        f" It is **more likely** than {join_with_and(other_names)} "
                        f"based on stronger alignment in {join_with_and(self.matched_fields[:3])}."
                    )
                else:
                    comparison = (
                        f" It is **less likely** than {join_with_and(other_names)} "
                        f"due to differences in {join_with_and(self.mismatched_fields[:3])}."
                    )

        return f"{intro} {summary}, the isolate most closely resembles **{self.genus}**. {confidence_text}{comparison}"


# ------------------------------------------------------------
# Bacteria Identifier
# ------------------------------------------------------------

class BacteriaIdentifier:
    """
    Main engine to match bacterial genus based on biochemical & morphological data.
    """

    def __init__(self, db: pd.DataFrame):
        self.db: pd.DataFrame = db.fillna("")
        self.db_columns = list(self.db.columns)

    # ---------- Field comparison ----------

    def compare_field(self, db_val: Any, user_val: Any, field_name: str) -> int:
        """
        Compare one test field between database and user input.
        Returns:
          +1  match
          -1  mismatch
           0  unknown / ignored
        Return -999 to indicate a hard exclusion (stop comparing this genus).
        """
        if user_val is None:
            return 0

        user_str = str(user_val).strip()
        if user_str == "" or user_str.lower() == "unknown":
            return 0  # ignore unknown/empty

        db_str = str(db_val).strip()
        db_l = db_str.lower()
        user_l = user_str.lower()

        hard_exclusions = {"Gram Stain", "Shape", "Spore Formation"}

        # Split multi-value fields on ; or / or ,
        db_options = [p.strip().lower() for p in re.split(r"[;/,]", db_str) if p.strip()]
        user_options = [p.strip().lower() for p in re.split(r"[;/,]", user_str) if p.strip()]

        # "variable" logic: if either is variable, don't penalize
        if "variable" in db_options or "variable" in user_options:
            return 0

        # Growth Temperature as range "low//high", user enters single numeric or similar
        if field_name == "Growth Temperature":
            try:
                if "//" in db_str:
                    low_s, high_s = db_str.split("//", 1)
                    low = float(low_s)
                    high = float(high_s)
                    # user may have given "37//37" or "37" etc.
                    if "//" in user_str:
                        ut = float(user_str.split("//", 1)[0])
                    else:
                        ut = float(user_str)
                    if low <= ut <= high:
                        return 1
                    else:
                        return -1
            except Exception:
                return 0

        # Flexible overlap match
        match_found = False
        for u in user_options:
            for d in db_options:
                if not d or not u:
                    continue
                if u == d:
                    match_found = True
                    break
                if u in d or d in u:
                    match_found = True
                    break
            if match_found:
                break

        if match_found:
            return 1

        if field_name in hard_exclusions:
            return -999  # hard mismatch
        return -1

    # ---------- Next-test suggestions ----------

    def suggest_next_tests(
        self,
        top_results: List[IdentificationResult],
        user_input: Dict[str, Any],
        max_tests: int = 3,
    ) -> List[str]:
        """
        Suggest tests that best differentiate top matches and haven't
        already been entered or marked 'Unknown' by the user.
        """
        if not top_results:
            return []

        # Only consider first 3–5 genera
        top_names = {r.genus for r in top_results[:5]}
        varying_fields: List[str] = []

        for field in self.db_columns:
            if field == "Genus":
                continue

            # Skip fields user already filled with a known value
            u_val = user_input.get(field, "")
            if isinstance(u_val, str) and u_val.lower() not in {"", "unknown"}:
                continue

            # Check if this field differs meaningfully between top genera
            values_for_field = set()
            for _, row in self.db.iterrows():
                g = row.get("Genus", "")
                if g in top_names:
                    v = str(row.get(field, "")).strip().lower()
                    if v:
                        values_for_field.add(v)

            if len(values_for_field) > 1:
                varying_fields.append(field)

        # simple deterministic: take first few
        return varying_fields[:max_tests]

    # ---------- Main identification routine ----------

    def identify(self, user_input: Dict[str, Any]) -> List[IdentificationResult]:
        """
        Compare user input to database and rank possible genera.
        Integrates extended signals when available.
        """

        results: List[IdentificationResult] = []
        total_fields_possible = len([c for c in self.db_columns if c != "Genus"])

        # Pre-compute extended scores if extended_reasoner is available
        extended_scores: Dict[str, float] = {}
        extended_explanation: str = ""

        if HAS_EXTENDED_REASONER:
            try:
                ranked_ext, explanation = score_genera_from_extended(user_input)
                extended_explanation = explanation or ""
                for genus, score in ranked_ext:
                    extended_scores[str(genus)] = float(score)
            except Exception:
                extended_scores = {}
                extended_explanation = ""

        for _, row in self.db.iterrows():
            genus = str(row.get("Genus", "")).strip()
            if not genus:
                continue

            total_score = 0
            matched_fields: List[str] = []
            mismatched_fields: List[str] = []
            reasoning_factors: Dict[str, Any] = {}
            total_fields_evaluated = 0

            hard_excluded = False

            for field in self.db_columns:
                if field == "Genus":
                    continue

                db_val = row.get(field, "")
                user_val = user_input.get(field, "")

                score = self.compare_field(db_val, user_val, field)

                if user_val is not None and str(user_val).strip() != "" and str(user_val).strip().lower() != "unknown":
                    total_fields_evaluated += 1

                if score == -999:
                    hard_excluded = True
                    total_score = -999
                    break
                elif score == 1:
                    total_score += 1
                    matched_fields.append(field)
                    reasoning_factors[field] = user_val
                elif score == -1:
                    total_score -= 1
                    mismatched_fields.append(field)

            if hard_excluded:
                continue  # skip this genus entirely

            extra_notes = str(row.get("Extra Notes", "")).strip() if "Extra Notes" in row else ""

            r = IdentificationResult(
                genus=genus,
                total_score=total_score,
                matched_fields=matched_fields,
                mismatched_fields=mismatched_fields,
                reasoning_factors=reasoning_factors,
                total_fields_evaluated=total_fields_evaluated,
                total_fields_possible=total_fields_possible,
                extra_notes=extra_notes,
            )

            # Attach extended score if available
            if genus in extended_scores:
                r.extended_score = extended_scores[genus]
                r.extended_explanation = extended_explanation

            results.append(r)

        # Sort by core score descending
        results.sort(key=lambda r: r.total_score, reverse=True)

        # Suggest next tests for top few
        if results:
            next_tests = self.suggest_next_tests(results[:5], user_input)
            next_tests_str = ", ".join(next_tests) if next_tests else ""
            for r in results[:5]:
                r.reasoning_factors["next_tests"] = next_tests_str

        # Return top 10
        return results[:10]
