# AI Engineering Prep

A hands-on AI engineering prep project for building a production-shaped AI assistant in Python.

The project starts with structured LLM responses, then adds a local retrieval-augmented generation pipeline, retrieval evals, tool-calling, and mini-agent behavior.

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
- Runs an agent mode that lets the model choose between document search and calculation tools.
- Traces tool calls made by the agent.
- Runs basic agent tool-routing evals from `evals/agent_dataset.json`.

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
  tools.py           # Tool definitions and execution
  vector_store.py    # Qdrant-backed vector store

data/docs/           # Source documents for RAG
data/qdrant/         # Generated local Qdrant storage

evals/
  agent_dataset.json # Agent tool-routing eval cases
  rag_dataset.json   # Retrieval eval cases
  run_agent.py       # Agent eval runner
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

Run the tool-calling agent:

```bash
uv run python -m app.main "What is sqrt(144) and what is this project trying to build?" --agent
```

Agent mode gives the model access to two tools:

- `search_docs`: searches the local indexed documents and returns relevant passages.
- `calculate`: evaluates simple arithmetic expressions.

The agent loops until the model stops requesting tool calls or reaches the maximum tool-iteration limit. Each completed response still returns the same structured `AssistantResponse` shape as the non-agent path.

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

## Run Agent Evals

```bash
uv run python -m evals.run_agent
```

The agent eval runner sends each prompt through `complete_with_tools`, compares the actual tool calls against `expected_tools`, and reports accuracy, tool precision, and tool recall.

The current dataset covers calculator-only, document-search-only, no-tool, and compound search-plus-calculation cases.

## Current Status

- [x] Project bootstrap and configuration
- [x] Structured LLM output with Pydantic
- [x] Basic RAG ingestion and retrieval
- [x] Local retrieval guardrails
- [x] Basic retrieval evals
- [x] Tool calling and mini-agent behavior
- [x] Basic agent tool-routing evals
- [ ] Observability improvements
- [ ] Docker support
- [ ] Project polish
