# AI Engineering Prep

A hands-on AI engineering prep project for building a production-shaped AI assistant in Python.

The project starts with structured LLM responses, then adds a local retrieval-augmented generation pipeline, retrieval evals, and eventually tool-calling and mini-agent behavior.

The long-term goal is to build an AI Technical Discovery Assistant that can turn project notes, architecture decisions, technical documents, and tickets into implementation-ready outputs such as specs, acceptance criteria, risk lists, and test scenarios.

## What It Does Today

- Loads configuration from `.env` with Pydantic settings.
- Calls OpenAI models through a small LLM client.
- Parses model output into a typed `AssistantResponse` schema.
- Loads local Markdown and text documents from `data/docs`.
- Splits documents into overlapping chunks.
- Creates embeddings for document chunks.
- Persists embeddings in a local Qdrant vector store.
- Retrieves relevant chunks from Qdrant by cosine similarity.
- Filters weak retrieval matches with `RETRIEVAL_MIN_SCORE`.
- Answers with retrieved context by default.
- Refuses grounded answers when no retrieved context passes the threshold.
- Runs basic retrieval evals from `evals/rag_dataset.json`.

## Project Structure

```text
app/
  config.py          # Environment/settings loading
  documents.py       # Document loading and chunking
  embeddings.py      # OpenAI embedding client
  index.py           # Shared retrieval result dataclasses
  ingest.py          # Builds the local Qdrant collection
  llm.py             # OpenAI LLM client and prompt construction
  main.py            # CLI entrypoint
  schemas.py         # Structured response schemas
  vector_store.py    # Qdrant-backed vector store

data/docs/           # Source documents for RAG
data/qdrant/         # Generated local Qdrant storage

evals/
  rag_dataset.json   # Retrieval eval cases
  run_rag.py         # RAG retrieval eval runner

tests/               # Unit tests
```

## Setup

Install dependencies:

```bash
uv sync --extra dev
```

Create your local environment file:

```bash
cp .env.example .env
```

Then set your OpenAI API key in `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
```

Optional settings:

```bash
OPENAI_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DOCS_PATH=data/docs
QDRANT_PATH=data/qdrant
QDRANT_COLLECTION=documents
EMBEDDING_DIM=1536
RETRIEVAL_TOP_K=4
RETRIEVAL_MIN_SCORE=0.25
```

## Build The RAG Index

Before asking document-grounded questions, ingest the local docs:

```bash
uv run python -m app.ingest
```

This reads files from `data/docs`, chunks them, creates embeddings, recreates the configured Qdrant collection, and upserts the chunks into local Qdrant storage at `data/qdrant`.

## Run The App

Show the current configuration:

```bash
uv run python -m app.main
```

Ask a RAG-backed question:

```bash
uv run python -m app.main "What is this project trying to build?"
```

Show retrieved context before the model answers:

```bash
uv run python -m app.main "What are the basic steps of this RAG pipeline?" --show-context
```

Bypass retrieval:

```bash
uv run python -m app.main "Explain structured outputs" --no-rag
```

Allow a general LLM answer when no local context is found:

```bash
uv run python -m app.main "What is the best lasagna in Turin?" --allow-llm-general
```

## Run Tests

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

## Run Retrieval Evals

```bash
uv run python -m evals.run_rag
```

The eval runner embeds each eval question, searches the local Qdrant collection, applies the retrieval threshold, and checks whether each case found or rejected context as expected.

The eval dataset currently includes positive cases for project overview, structured outputs, and RAG steps, plus negative cases for unrelated restaurant and weather questions.

## Current Status

- [x] Project bootstrap and configuration
- [x] Structured LLM output with Pydantic
- [x] Basic RAG ingestion and retrieval
- [x] Local retrieval guardrails
- [x] Basic retrieval evals
- [ ] Tool calling and mini-agent behavior
- [ ] Observability improvements
- [ ] Docker support
- [ ] Project polish
