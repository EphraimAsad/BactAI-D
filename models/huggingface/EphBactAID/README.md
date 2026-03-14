---
license: apache-2.0
base_model:
- google/flan-t5-base
pipeline_tag: feature-extraction
library_name: transformers
tags:
- biology
language:
- en
---
### Model Card: Core Schema Parsing LLM (Microbiology)
## Model Overview

This model is a domain-adapted sequence-to-sequence language model designed to parse free-text microbiology phenotype descriptions into a structured core schema of laboratory test results and traits.

The model is intended to augment deterministic rule-based and extended parsers by recovering fields that may be missed due to complex phrasing, implicit descriptions, or uncommon linguistic constructions. It is not designed to operate as a standalone classifier or diagnostic system.

## Base Model

Base architecture: google/flan-t5-base

Model type: Encoder–decoder (Seq2Seq), instruction-tuned

The FLAN-T5 base model was selected due to its strong instruction-following behaviour, stability during fine-tuning, and suitability for structured text generation tasks on limited hardware.

## Training Data

The model was fine-tuned on 8,700 curated microbiology phenotype examples, each consisting of:

A free-text phenotype description

A deterministic target serialization of core schema fields and values

Data preprocessing:

The name field and all non-core schema fields were explicitly removed to prevent label leakage.

Target outputs were serialized deterministically using sorted schema keys (Field: Value format).

Inputs and targets were constrained to schema-relevant content only.

The dataset was split 80/20 into training and validation subsets.

# Training Procedure

- Epochs: 3

- Optimizer: AdamW (default Hugging Face Trainer)

- Learning rate: 1e-5

# Batching:

- Per-device batch size: 1

- Gradient accumulation: 8 (effective batch size = 8)

- Sequence lengths:

- Max input length: 2048 tokens

- Max output length: 2048 tokens

# Precision:

- bf16 on supported hardware (A100), otherwise fp16

- Stability measures:

- Gradient checkpointing enabled

- Gradient clipping (max_grad_norm = 1.0)

- Warmup ratio of 0.03

- The model was trained using the Hugging Face Trainer API and saved after completion of all epochs.

## Intended Use

This model is intended for:

Structured parsing of microbiology phenotype text into predefined schema fields

Use as a third-stage parser alongside rule-based and extended parsers

Supporting downstream deterministic scoring, ranking, and retrieval systems

Not intended for:

Standalone clinical diagnosis

Autonomous decision-making

Use without additional validation layers

## Integration Context

In production, the model is used as a fallback and recovery mechanism within a hybrid parsing pipeline:

- Rule-based parser (high precision)

- Extended parser (schema-aware)

- LLM parser (coverage and robustness)

Outputs are reconciled and validated downstream before being used for identification or explanation.

## Limitations

Performance depends on coverage of the training schema and cannot generalize beyond it.

The model may hallucinate field values if used outside its intended constrained pipeline.

It is sensitive to extreme deviations in input style or unsupported terminology.

## Ethical and Safety Considerations

The model does not provide medical advice or diagnoses.

Outputs should always be reviewed in conjunction with deterministic logic and domain expertise.

Training data was curated to minimize leakage and unintended inference.

## Author

Developed and fine-tuned by Zain Asad as part of the BactAI-D project.