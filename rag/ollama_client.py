# rag/ollama_client.py
# ============================================================
# Ollama client wrapper for RAG generation
#
# Provides a simple interface to Ollama API with:
# - Connection checking
# - Retry logic
# - Timeout handling
# - Model availability checking
# ============================================================

from __future__ import annotations

import os
import time
from typing import Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DEFAULT_MODEL = os.getenv("BACTAI_OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_TIMEOUT = int(os.getenv("BACTAI_OLLAMA_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("BACTAI_OLLAMA_RETRIES", "2"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

_client: Optional["ollama.Client"] = None


# ------------------------------------------------------------
# CLIENT MANAGEMENT
# ------------------------------------------------------------

def get_client() -> "ollama.Client":
    """Get or create Ollama client instance."""
    global _client
    if not OLLAMA_AVAILABLE:
        raise RuntimeError(
            "Ollama package not installed. Run: pip install ollama"
        )
    if _client is None:
        _client = ollama.Client(host=OLLAMA_HOST)
    return _client


def is_ollama_running() -> bool:
    """Check if Ollama server is running and accessible."""
    if not OLLAMA_AVAILABLE:
        return False
    try:
        client = get_client()
        client.list()
        return True
    except Exception:
        return False


def _extract_model_names(models_response) -> list[str]:
    """Extract model names from Ollama list response (handles both old and new API)."""
    try:
        # New API: models_response.models is a list of Model objects with .model attribute
        if hasattr(models_response, 'models'):
            model_list = models_response.models
            names = []
            for m in model_list:
                # Try .model attribute first (new API), then .name, then dict access
                if hasattr(m, 'model'):
                    names.append(m.model)
                elif hasattr(m, 'name'):
                    names.append(m.name)
                elif isinstance(m, dict):
                    names.append(m.get('model', m.get('name', '')))
            return names
        # Old API: dict with "models" key
        elif isinstance(models_response, dict):
            return [m.get('model', m.get('name', '')) for m in models_response.get('models', [])]
        return []
    except Exception:
        return []


def is_model_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if the specified model is available locally."""
    if not OLLAMA_AVAILABLE:
        return False
    try:
        client = get_client()
        models = client.list()
        model_names = _extract_model_names(models)
        # Check exact match or partial match (e.g., "llama3.2:3b" in "llama3.2:3b-instruct")
        return any(model in name or name.startswith(model.split(":")[0]) for name in model_names)
    except Exception:
        return False


def get_available_models() -> list[str]:
    """Get list of locally available models."""
    if not OLLAMA_AVAILABLE:
        return []
    try:
        client = get_client()
        models = client.list()
        return _extract_model_names(models)
    except Exception:
        return []


# ------------------------------------------------------------
# GENERATION
# ------------------------------------------------------------

def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """
    Generate text using Ollama.

    Args:
        prompt: The input prompt
        model: Model name (default: llama3.2:3b)
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Returns:
        Generated text string

    Raises:
        RuntimeError: If Ollama is not available or generation fails
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama package not installed")

    client = get_client()
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "repeat_penalty": 1.2,
                },
            )
            return response.get("response", "").strip()

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(1)  # Brief delay before retry
                continue
            break

    raise RuntimeError(f"Ollama generation failed after {MAX_RETRIES + 1} attempts: {last_error}")


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    """
    Chat-style generation using Ollama.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated response text
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama package not installed")

    client = get_client()
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "repeat_penalty": 1.2,
                },
            )
            return response.get("message", {}).get("content", "").strip()

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            break

    raise RuntimeError(f"Ollama chat failed after {MAX_RETRIES + 1} attempts: {last_error}")


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

def health_check() -> dict:
    """
    Perform a comprehensive health check.

    Returns:
        Dict with status information
    """
    result = {
        "ollama_installed": OLLAMA_AVAILABLE,
        "server_running": False,
        "model_available": False,
        "model_name": DEFAULT_MODEL,
        "available_models": [],
        "error": None,
    }

    if not OLLAMA_AVAILABLE:
        result["error"] = "Ollama package not installed"
        return result

    try:
        result["server_running"] = is_ollama_running()
        if result["server_running"]:
            result["available_models"] = get_available_models()
            result["model_available"] = is_model_available(DEFAULT_MODEL)
            if not result["model_available"]:
                result["error"] = f"Model '{DEFAULT_MODEL}' not found. Run: ollama pull {DEFAULT_MODEL}"
    except Exception as e:
        result["error"] = str(e)

    return result
