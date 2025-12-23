# rag/rag_embedder.py
# ============================================================
# Embedding utilities for RAG (knowledge base + queries)
# Uses a SentenceTransformer model for dense embeddings.
# ============================================================

from __future__ import annotations

import os
import json
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

_model: SentenceTransformer | None = None


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------

def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


# ------------------------------------------------------------
# EMBEDDING
# ------------------------------------------------------------

def embed_text(text: str, normalize: bool = True) -> np.ndarray:
    """
    Embed a single piece of text.
    Returns a 1D numpy array (MPNet: 768-dim).
    """
    model = get_embedder()
    emb = model.encode(
        [text],
        show_progress_bar=False,
        normalize_embeddings=normalize,
    )
    return emb[0]


def embed_texts(texts: List[str], normalize: bool = True) -> np.ndarray:
    """
    Embed a list of strings -> (N, D) numpy array.
    """
    model = get_embedder()
    return model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    )


# ------------------------------------------------------------
# INDEX LOADING
# ------------------------------------------------------------

def load_kb_index(path: str = "data/rag/index/kb_index.json") -> Dict[str, Any]:
    """
    Load the RAG knowledge base index JSON.

    Expected format:
    {
      "version": int,
      "model_name": str,
      "records": [
        {
          "id": str,
          "genus": str,
          "species": str | null,
          "level": "genus" | "species",
          "chunk_id": int,
          "source_file": str,
          "text": str,
          "embedding": [float, ...]
        }
      ]
    }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"KB index not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index_model = data.get("model_name")
    if index_model != EMBEDDING_MODEL_NAME:
        raise ValueError(
            f"KB index built with '{index_model}', "
            f"but current embedder is '{EMBEDDING_MODEL_NAME}'. "
            "Rebuild the index."
        )

    # Convert embeddings to numpy arrays
    for rec in data.get("records", []):
        if isinstance(rec.get("embedding"), list):
            rec["embedding"] = np.array(rec["embedding"], dtype="float32")

    return data