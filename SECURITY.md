# Security and Public Data

## Secrets

Do not commit API keys, access tokens, private keys, passwords, or `.env` files.
Runtime credentials must be supplied through environment variables.

Current code expects secrets only through environment variables, for example:

- `HF_TOKEN`
- `BACTAI_OLLAMA_MODEL`

## Public Data Policy

Everything committed to this repository must be safe to treat as public.
That includes source code, docs, configuration, curated reference data, trained model artifacts, and generated RAG indexes.

The current repository intentionally publishes:

- `data/bacteria_db.xlsx`
- `data/rag/knowledge_base/`
- `data/rag/index/kb_index.json`
- `models/`

`data/rag/index/kb_index.json` includes public reference text chunks and embeddings derived from the committed knowledge base. Do not place private or regulated content into the knowledge-base source files or generated indexes.

## Review Guidance

Before pushing:

- remove personal contact data unless it is intentionally public project metadata
- keep credentials in environment variables only
- treat generated indexes and exported datasets as public artifacts if they remain tracked
- run CI and resolve any secret-scan failures before merging
