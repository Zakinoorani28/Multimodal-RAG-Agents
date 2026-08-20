# Multimodal RAG — Attention Is All You Need

AI Season Task 5 — Multimodal RAG system that answers questions
about the Transformer paper using text, tables, and figure understanding.

## Models Used

- **Generation + Vision**: gemini-2.5-flash
- **Embeddings**: models/text-embedding-004
- **Vector Store**: ChromaDB (cosine similarity, persistent)

## Requirements checklist satisfied

✓ Gemini API with key from environment variable
✓ Attention Is All You Need paper as source
✓ Extracts: body text, headings, tables, figures
✓ Gemini Vision describes all figures
✓ All content embedded in shared vector space
✓ Semantic retrieval step
✓ Grounded answer generation
✓ Sample queries hit table, figure, and text modalities
✓ Query + context + answer saved to outputs/sample_outputs.json

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Place the PDF

`data/attention.pdf` ← put the paper here

## Run

```bash
# Step 1: Build knowledge base
python ingest.py

# Step 2: Run sample queries (creates outputs/sample_outputs.json)
python sample_queries.py

# Step 3: Launch UI
streamlit run app/main.py
```

## What to submit

- `app/` folder
- `ingest.py`
- `sample_queries.py`
- `requirements.txt`
- `outputs/sample_outputs.json`
- `.env.example`
- `README.md`

Do NOT submit: `.env`, `chroma_db/`, `data/figures/`
