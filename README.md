🧫 BactAI-D
Hybrid AI System for Microbiology Phenotype Identification

- BactAI-D is an end-to-end, production-oriented AI system for identifying bacterial genera and species from free-text microbiology phenotype descriptions.
- It combines rule-based parsing, schema-driven learning, machine learning, and retrieval-augmented generation (RAG) into a unified, confidence-aware identification pipeline designed for real laboratory data.

✨ Key Capabilities

🔹 Hybrid Parsing Architecture
- Rule Parser — deterministic extraction of core microbiology fields
- Extended Parser — schema-aware extraction of advanced tests (e.g. pigments, motility types, TSI, NaCl tolerance)
- Tri-Fusion Engine — merges rule, extended, and optional LLM parsing safely (LLM off by default)
- Parsing is explicit-only: no hallucination, no inference, ML-safe by design.


🔹 Schema-Driven & Self-Trainable
- Supports dynamic extension of supported tests via curated schema files
- Can train itself on new fields using structured “gold test” datasets
- Parser accuracy evaluated automatically via internal evaluation tooling
- This allows the system to grow without rewriting core logic.


🔹 ML Genus Prediction (XGBoost)
- Trained on parsed phenotypic features
- ~8300 Test profiles, with over 300,000 total fields to train upon.
- Outputs calibrated genus probabilities
- Integrated with rule-based confidence via hybrid weighting


🔹 Unified Confidence Engine
Combines:
- Tri-Fusion rule confidence
- ML genus probabilities
- Applies hard confidence gates to prevent overconfidence
- Uses decision-safe confidence bands:
  - Low Discrimination
  - Acceptable Identification
  - Good Identification
  - Excellent Identification
- Only the top-ranked genus may receive a positive confidence label.


🔹 Species Scoring (Within Genus)
- Species prediction is constrained to the top-ranked genus
- Uses phenotype similarity scoring (not free inference)
- Prevents cross-genus hallucination


🔹 Retrieval-Augmented Generation (RAG)
- Genus-specific knowledge base
- LLM explanations grounded only in retrieved microbiology context
- Includes:
  - Phenotypic reasoning
  - Supporting traits
  - Best-match species summary
  - RAG is explanatory — never authoritative.


🔹 Human-Safe Decision Table
- Top-5 output includes:
  - Probability % (normalized within top-5)
  - Human-friendly odds (“1 in X”)
  - Decision confidence (rank-1 only)
  - Designed to support interpretation, not automation.


🧠 System Architecture (High Level)

Phenotype Text
      ↓
      
Rule Parser
      ↓

Extended Parser (Schema-Aware)
      ↓

Tri-Fusion Merge
      ↓

ML Genus Prediction
      ↓

Unified Confidence Scoring
      ↓

Top-5 Decision Table
      ↓

RAG Explanation (Genus + Species)



🧪 Training & Extensibility
- BactAI-D includes tooling to:
  - Evaluate parser accuracy
  - Train on curated gold test datasets
  - Learn new schema fields
  - Retrain ML models
  - Rebuild the RAG index
- This enables continuous improvement without architectural changes.


🖥️ Demo (Hugging Face Spaces)
A live interactive demo is available on Hugging Face Spaces: https://huggingface.co/spaces/EphAsad/BactAID-Demo

- LLM parsing disabled by default (safe deployment)
- Full debug outputs available
- Recruiter-friendly UI with explainability


⚠️ Disclaimer

This project is not a medical diagnostic device.
It is intended for research, education, and decision support only.


🚀 Future Directions (Planned)
- Genus-specific confirmatory test recommendations
- Progressive diagnostic workflows
- Multi-genus explanatory comparison
- Expanded species-level reasoning
- Confidence calibration analysis


👤 Author

Zain Asad
Microbiology × Applied AI
Built as an independent research and engineering project.
