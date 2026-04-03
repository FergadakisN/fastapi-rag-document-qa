# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Exercise 7 for DTU's NLP course. Build a FastAPI question-answering service over a PDF textbook using RAG (Retrieval-Augmented Generation). Full requirements are in `ask-pdf.md`.

## Running the service

```bash
# Install dependencies
pip install fastapi uvicorn openai python-dotenv numpy

# Start the server (from active/)
uvicorn main:app --reload

# Query the endpoint
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the title of the book?"}'
```

## Environment

Credentials live in `active/.env`:

```
CAMPUSAI_API_KEY=...
CAMPUSAI_CHAT_MODEL=Gemma 3 (Chat)
CAMPUSAI_API_URL=https://chat.campusai.compute.dtu.dk/api/v1
```

The CampusAI API is OpenAI-compatible — use the `openai` Python package with `base_url` and `api_key` overrides. Requires DTU network (Eduroam or VPN).

Both models share the same client and API key:
- **`Nomic Embed Text`** — embedding model for indexing chunks and embedding queries
- **`Gemma 3 (Chat)`** — chat model for generating answers from retrieved chunks

## Source material

| File                                        | Description                                                                        |
| ------------------------------------------- | ---------------------------------------------------------------------------------- |
| `Nielsen2025Natural-2026-03-20.md`          | Best source for RAG — 12,192 lines, well-structured Markdown with chapter headings |
| `Nielsen2025Natural-2026-03-20.pdf.tei.xml` | TEI XML from GROBID — richer metadata (author, references)                         |
| `Nielsen2025Natural-2026-03-20.txt`         | Plain text fallback                                                                |

Prefer the `.md` file for chunking — headings (`## Chapter N`, section titles) make natural chunk boundaries.

## Architecture

RAG pipeline in `active/main.py`:

1. **Chunking** — Split `.md` by headings or fixed token windows (~500 tokens, ~100 token overlap). Done once at startup.
2. **Indexing** — Embed all chunks with `nomic-embed-text` via CampusAI embeddings API; store as a numpy matrix.
3. **Retrieval** — Embed the incoming question, compute cosine similarity against the chunk matrix, return top-k (k≈5) chunks.
4. **Generation** — Construct a prompt with the retrieved chunks + question; call the LLM (`Gemma 3 (Chat)`) to produce `answer` and `followup_questions`.

## FastAPI contract

```
POST /api/v1/ask
Body:  { "question": "string" }
Returns: { "answer": "string", "followup_questions": ["string", ...] }
```

## Deliverable

```bash
git archive -o latest.zip HEAD
```
