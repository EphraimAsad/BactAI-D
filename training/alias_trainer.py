# training/alias_trainer.py  
# ------------------------------------------------------------  
# Stage 10B - Alias Trainer  
#  
# Learns field/value synonyms from gold tests by comparing:  
#   - expected values (gold standard)  
#   - parsed values (rules + extended)  
#  
# Outputs:  
#   - Updated alias_maps.json  
#  
# This is the core intelligence that allows BactAI-D  
# to understand variations in microbiology language.  
# ------------------------------------------------------------  
  
import json  
import os  
from collections import defaultdict  
  
from engine.parser_rules import parse_text_rules  
from engine.parser_ext import parse_text_extended  
  
  
GOLD_PATH = "training/gold_tests.json"  
ALIAS_PATH = "data/alias_maps.json"  
  
  
def normalise(s):  
    if s is None:  
        return ""  
    return str(s).strip().lower()  
  
  
def learn_aliases():  
    """  
    Learns synonym mappings from gold tests.  
    """  
    if not os.path.exists(GOLD_PATH):  
        return {"error": f"Gold tests missing: {GOLD_PATH}"}  
  
    with open(GOLD_PATH, "r", encoding="utf-8") as f:  
        gold = json.load(f)  
  
    # Load or create alias map  
    if os.path.exists(ALIAS_PATH):  
        with open(ALIAS_PATH, "r", encoding="utf-8") as f:  
            alias_maps = json.load(f)  
    else:  
        alias_maps = {}  
  
    # Track suggestions  
    suggestions = defaultdict(lambda: defaultdict(int))  
  
    # ------------------------------------------------------------  
    # Compare expected vs parsed for all tests  
    # ------------------------------------------------------------  
    for test in gold:  
        text = test.get("input", "")  
        expected = test.get("expected", {})  
  
        rules = parse_text_rules(text).get("parsed_fields", {})  
        ext = parse_text_extended(text).get("parsed_fields", {})  
  
        # merge deterministic parsers  
        merged = dict(rules)  
        for k, v in ext.items():  
            if v != "Unknown":  
                merged[k] = v  
  
        # now compare with expected  
        for field, exp_val in expected.items():  
            exp_norm = normalise(exp_val)  
            got_norm = normalise(merged.get(field, "Unknown"))  
  
            # Skip correct matches  
            if exp_norm == got_norm:  
                continue  
  
            # Skip unknown expected  
            if exp_norm in ["", "unknown"]:  
                continue  
  
            # Mismatched → candidate alias  
            if got_norm not in ["", "unknown"]:  
                suggestions[field][got_norm] += 1  
  
    # ------------------------------------------------------------  
    # Convert suggestions into alias mappings  
    # ------------------------------------------------------------  
    alias_updates = {}  
  
    for field, values in suggestions.items():  
        # ignore fields with tiny evidence  
        for wrong_value, count in values.items():  
            if count < 2:  
                continue  # avoid noise  
              
            # add/update alias  
            if field not in alias_maps:  
                alias_maps[field] = {}  
  
            # map wrong_value → expected canonical version  
            # canonical version is the most common value in gold_tests for that field  
            canonical = None  
            # determine canonical  
            field_values = [normalise(t["expected"][field]) for t in gold if field in t["expected"]]  
            if field_values:  
                # most common expected value  
                canonical = max(set(field_values), key=field_values.count)  
  
            if canonical:  
                alias_maps[field][wrong_value] = canonical  
                alias_updates[f"{field}:{wrong_value}"] = canonical  
  
    # ------------------------------------------------------------  
    # Save alias maps  
    # ------------------------------------------------------------  
    with open(ALIAS_PATH, "w", encoding="utf-8") as f:  
        json.dump(alias_maps, f, indent=2)  
  
    return {  
        "ok": True,  
        "updated_aliases": alias_updates,  
        "total_updates": len(alias_updates),  
        "alias_map_path": ALIAS_PATH,  
    }
