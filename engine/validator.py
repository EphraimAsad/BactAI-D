# engine/validator.py
# ---------------------------------
# Placeholder for logical validation layer

def validate_record(parsed: dict) -> dict:
    """
    Later: check for contradictions, invalid values,
    and normalize to schema.
    """
    parsed.setdefault("validation_notes", [])
    parsed["validation_notes"].append("Validator not yet implemented.")
    return parsed
