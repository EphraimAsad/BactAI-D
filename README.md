# BactAI-D: AI-Powered Bacterial Identification System

<p align="center">
  <img src="static/eph.jpeg" alt="BactAI-D Logo" width="150"/>
</p>

<p align="center">
  <strong>Intelligent bacterial genus and species identification from phenotypic descriptions</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#api">API</a>
</p>

---

## Overview

BactAI-D (Bacterial AI-Diagnostics) is a sophisticated AI-powered system designed for microbiology laboratories to identify bacterial genera and species from phenotypic test descriptions. It combines multiple AI techniques including:

- **Tri-Parser Fusion** - Three complementary parsing approaches for robust text interpretation
- **Machine Learning** - XGBoost classifier trained on bacterial phenotype patterns
- **RAG (Retrieval-Augmented Generation)** - Knowledge base-backed explanations powered by Ollama
- **Deterministic Scoring** - Rule-based validation ensuring reliable results

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Natural Language Input** | Enter phenotype descriptions in plain English |
| **Multi-Method Identification** | Combines rule-based, ML, and AI approaches |
| **Confidence Scoring** | Clear confidence bands (Excellent/Good/Acceptable/Low) |
| **Species-Level Matching** | Narrows down to specific species within genera |
| **AI-Powered Explanations** | Detailed reasoning via Ollama LLM integration |
| **Knowledge Base** | 100+ bacterial genera with comprehensive phenotype data |

### Identification Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                       │
│  "Gram positive cocci in clusters, catalase positive, coagulase positive"│
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      TRI-PARSER FUSION                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ Rule Parser  │  │Extended Parser│  │  LLM Parser  │                   │
│  │   (Regex)    │  │ (Biochemical) │  │ (EphBactAID) │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
│           └──────────────┼──────────────┘                               │
│                    Weighted Voting                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
         ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
         │  Database    │ │   XGBoost    │ │  Diagnostic  │
         │  Identifier  │ │  Predictor   │ │   Anchors    │
         └──────────────┘ └──────────────┘ └──────────────┘
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      UNIFIED RANKING                                     │
│  Combines scores → Applies confidence bands → Ranks top 5 genera        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RAG EXPLANATION (Ollama)                              │
│  Retrieves knowledge → Shapes context → Generates natural explanation   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                           │
│  Genus: Staphylococcus (98.28%) - Good Identification                   │
│  Species: Staphylococcus aureus (100% match)                            │
│  Explanation: "This phenotype is indicative of Staphylococcus..."       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. **Python 3.10+** - [Download](https://python.org)
2. **Ollama** - [Download](https://ollama.com) (optional, for AI explanations)

### Installation

```bash
# Clone or extract BactAI-D
cd BactAID

# Install dependencies
pip install -r backend/requirements.txt

# Download Ollama model (optional but recommended)
ollama pull llama3.2:3b
```

### Running

**Windows:**
```
Double-click BactAID_Launcher.bat
```

**Command Line:**
```bash
python backend/app.py
# Open browser to http://localhost:8000
```

## How It Works

### 1. Tri-Parser Fusion

BactAI-D uses three complementary parsers to extract structured data from free-text phenotype descriptions:

| Parser | Method | Strength |
|--------|--------|----------|
| **Rule Parser** | Regex patterns | Fast, precise for standard terminology |
| **Extended Parser** | Biochemical logic | Handles complex test interpretations |
| **LLM Parser** | Fine-tuned T5 model | Understands natural language variations |

The parsers vote on each field with learned reliability weights, producing a robust fused result.

### 2. Genus Prediction

Two parallel scoring methods:

- **Database Identifier**: Matches parsed fields against 122 bacterial records in the Excel database
- **XGBoost Classifier**: ML model trained on phenotype feature vectors

Results are combined using adaptive weighting based on ML confidence.

### 3. RAG Explanation System

For each top genus, the system:

1. **Retrieves** relevant knowledge from the genus-specific knowledge base
2. **Shapes** context by comparing parsed traits vs. reference traits
3. **Generates** a natural language explanation using Ollama (Llama 3.2)
4. **Falls back** to deterministic templates if Ollama is unavailable

### 4. Species Matching

Within each genus, species are ranked by:
- Expected field matches
- Species-specific markers
- Confidence scoring

## Project Structure

```
BactAID/
├── backend/
│   ├── app.py              # Flask REST API server
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/                # React source code
│   └── dist/               # Built frontend (served by Flask)
├── engine/
│   ├── bacteria_identifier.py   # Core scoring engine
│   ├── parser_fusion.py         # Tri-parser fusion
│   ├── parser_rules.py          # Rule-based parser
│   ├── parser_ext.py            # Extended biochemical parser
│   ├── parser_llm.py            # LLM parser (EphBactAID)
│   ├── genus_predictor.py       # XGBoost ML model
│   └── schema.py                # Field definitions
├── rag/
│   ├── rag_generator.py    # Ollama-powered explanation generator
│   ├── rag_retriever.py    # Knowledge base retrieval
│   ├── rag_embedder.py     # Semantic embeddings
│   ├── ollama_client.py    # Ollama API wrapper
│   └── species_scorer.py   # Species-level matching
├── scoring/
│   ├── overall_ranker.py       # Final ranking algorithm
│   └── diagnostic_anchors.py   # Special diagnostic rules
├── training/
│   ├── gold_trainer.py         # Training from gold tests
│   ├── field_weight_trainer.py # Parser weight learning
│   └── rag_index_builder.py    # Knowledge base indexing
├── data/
│   ├── bacteria_db.xlsx        # Main bacterial database
│   ├── rag/knowledge_base/     # Genus/species knowledge files
│   └── rag/index/              # Pre-built embeddings index
├── models/
│   ├── genus_xgb.json          # Trained XGBoost model
│   └── huggingface/            # Cached transformer models
└── static/
    └── eph.jpeg                # Logo
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/health` | GET | Server health check |
| `/api/ollama-status` | GET | Ollama connection status |
| `/api/identify` | POST | Main identification endpoint |

### Identification Request

```bash
POST /api/identify
Content-Type: application/json

{
  "text": "Gram positive cocci in clusters, catalase positive, coagulase positive",
  "use_llm": false
}
```

### Response

```json
{
  "top5_table": [
    ["Staphylococcus", "98.28%", "1 in 1", "Good Identification"],
    ["Rhodococcus", "0.43%", "1 in 250", "Low Discrimination"]
  ],
  "genus_cards": [
    {
      "genus": "Staphylococcus",
      "combined_percent": 98.28,
      "decision_band": "Good Identification",
      "rag_text": "KEY TRAITS:\n- Gram Stain: Positive\n...\nCONCLUSION:\n..."
    }
  ]
}
```

## Configuration

### Environment Variables

Sensitive values must be provided through environment variables and must not be committed. Public-data handling and secret hygiene are documented in [SECURITY.md](SECURITY.md).

| Variable | Default | Description |
|----------|---------|-------------|
| `BACTAI_OLLAMA_MODEL` | `llama3.2:3b` | Ollama model for RAG |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `BACTAI_LLM_FEWSHOT` | `0` | Few-shot examples for LLM parser |
| `BACTAI_RAG_GEN_LOG_INPUT` | `0` | Debug: log RAG prompts |
| `BACTAI_RAG_GEN_LOG_OUTPUT` | `0` | Debug: log RAG outputs |

## Technologies

| Component | Technology |
|-----------|------------|
| Backend | Flask 3.0, Python 3.11 |
| Frontend | React 18, Vite, Tailwind CSS |
| ML Models | XGBoost, Sentence Transformers |
| LLM | Ollama (Llama 3.2), HuggingFace Transformers |
| Database | Excel (pandas), FAISS vector index |

## Implementation Guide

### Laboratory Deployment Process

#### Phase 1: Technical Setup (IT Department)

```
Week 1: Infrastructure Preparation
├── Day 1-2: Verify system requirements on target workstations
├── Day 3-4: Install Python 3.10+ and Ollama
├── Day 5: Extract BactAI-D, install dependencies
└── Day 5: Run initial system test
```

#### Phase 2: Validation (Lab Supervisor)

```
Week 2: System Validation
├── Day 1: Test with 10 known organisms (ATCC strains)
├── Day 2-3: Document accuracy and discrepancies
├── Day 4: Adjust workflow based on findings
└── Day 5: Prepare training materials
```

#### Phase 3: Staff Training (All Personnel)

```
Week 3: Training Rollout
├── Session 1: System overview and startup procedures
├── Session 2: Data entry best practices
├── Session 3: Interpreting results and confidence bands
└── Session 4: Troubleshooting and QC procedures
```

### Standard Operating Procedure Summary

```
1. ISOLATE     → Pure culture required
2. OBSERVE     → Gram stain, morphology, colony characteristics
3. TEST        → Catalase, oxidase, additional biochemicals
4. ENTER       → Input all observations into BactAI-D
5. ANALYZE     → Click Identify, review results
6. INTERPRET   → Check confidence band and explanations
7. CONFIRM     → Perform confirmatory test if needed
8. DOCUMENT    → Record final identification
```

### Quality Control Requirements

| Frequency | Activity | Acceptance Criteria |
|-----------|----------|---------------------|
| Daily | Run QC organism | Top match correct, ≥90% confidence |
| Weekly | Test 3 diverse organisms | All correct, ≥85% confidence |
| Monthly | Full validation panel | ≥95% accuracy on 20 organisms |
| Annual | Complete system review | Document performance metrics |

### Multi-Workstation Deployment

**Recommended Architecture:**
```
                    ┌─────────────────┐
                    │  BactAI-D       │
                    │  Server         │
                    │  (Dedicated PC) │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │ Workstation │   │ Workstation │   │ Workstation │
    │     #1      │   │     #2      │   │     #3      │
    │  (Browser)  │   │  (Browser)  │   │  (Browser)  │
    └─────────────┘   └─────────────┘   └─────────────┘
```

**Server Setup:**
```bash
# On server machine
python backend/app.py
# Note the IP address (e.g., 192.168.1.100)

# On workstations, open browser to:
http://192.168.1.100:8000
```

## Training & Customization

BactAI-D includes tools for training and customization:

- **Gold Tests**: Add test cases to `data/gold_tests.json`
- **Parser Weights**: Run `/api/train-weights` to relearn parser reliability
- **Genus Model**: Run `/api/train-genus` to retrain XGBoost
- **Knowledge Base**: Add genus JSON files to `data/rag/knowledge_base/`

### Adding New Organisms

1. **Database Entry**
   ```
   Open: data/bacteria_db.xlsx
   Add row with: Genus, Species, Gram, Shape, Arrangement, tests...
   Save file
   ```

2. **Knowledge Base Entry** (for AI explanations)
   ```
   Create: data/rag/knowledge_base/NewGenus/genus.json
   Create: data/rag/knowledge_base/NewGenus/species.json
   Format: Follow existing genus examples
   ```

3. **Rebuild Index**
   ```
   Start BactAI-D → Training tab → Click "Build RAG Index"
   ```

## License

This project is proprietary software developed for microbiological research and clinical laboratory use.

## Support

For issues and feature requests, please contact the development team or open an issue on the repository.

---

<p align="center">
  <strong>BactAI-D</strong> - Bringing AI to Bacterial Identification
</p>
