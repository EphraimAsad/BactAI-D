title: BactKing
emoji: 💻
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
short_description: Hybrid microbiology identification with ML, rules, and RAG

🧫 BactAI-D — Hybrid Microbiology Identification System

BactAI-D is an applied AI system for phenotype-based bacterial genus and species identification, designed to balance deterministic microbiology rules, machine learning, and LLM-based explanation in a single, safety-aware pipeline.

Unlike end-to-end black-box models, BactAI-D uses a hybrid inference architecture:

Rule-based + extended schema parsing for explicit, ML-safe phenotype extraction

Tri-Fusion scoring to combine deterministic and probabilistic evidence

XGBoost genus prediction with confidence calibration

Unified ranking with decision gating (only rank-1 may be considered actionable)

RAG-based genus explanations constrained to post-decision interpretation (no hallucinated inference)

The system is built to expose uncertainty, not hide it — using confidence bands, rank-1 acceptance rules, and human-readable decision tables rather than forced predictions.

Intended use: educational, exploratory, and decision-support contexts in microbiology and applied AI research.
Not intended for clinical diagnosis or unsupervised laboratory decisions.

