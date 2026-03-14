from pathlib import Path

BASE_LOCAL = Path(
    r"C:\Users\adama\OneDrive\Documents\Zain\BactAID\models\huggingface"
)

MODELS = {
    "bactaid": {
        "hf_id": "EphAsad/EphBactAID",
        "local": BASE_LOCAL / "EphBactAID",
    },
    "bart": {
        "hf_id": "facebook/bart-large",
        "local": BASE_LOCAL / "bart-large",
    },
    "embedder": {
        "hf_id": "sentence-transformers/all-mpnet-base-v2",
        "local": BASE_LOCAL / "all-mpnet-base-v2",
    },
}
