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
- Tracks token usage and estimated cost for agent runs.
- Persists each CLI run as a trace in Postgres when the database is available.
- Runs basic agent tool-routing evals from `evals/agent_dataset.json`.
- Exposes the same document search and calculator capabilities through a small MCP server.

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
  observability.py   # Usage collection and cost estimation helpers
  schemas.py         # Structured response schemas
  tools.py           # Tool definitions and execution
  trace_store.py     # Postgres trace persistence
  vector_store.py    # Qdrant-backed vector store

data/docs/           # Source documents for RAG
data/qdrant/         # Generated local Qdrant storage

evals/
  agent_dataset.json # Agent tool-routing eval cases
  rag_dataset.json   # Retrieval eval cases
  run_agent.py       # Agent eval runner
  run_rag.py         # RAG retrieval eval runner

tests/               # Unit tests

docker-compose.yml   # Local Postgres service for trace storage
mcp_server.py        # FastMCP server exposing project tools
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
DATABASE_URL=postgresql://aiprep:aiprep@localhost:5433/aiprep
```

## Start Postgres

The project can persist run traces to a local Postgres database. Start it with Docker Compose:

```bash
docker compose up -d
```

This starts a `postgres:17` container on local port `5433` with the default credentials from `.env.example`.

The app uses `DATABASE_URL` to connect. If Postgres is unavailable, the CLI still answers normally and logs a trace-store warning instead of failing the run.

To stop the database:

```bash
docker compose down
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

## Observability And Traces

Each CLI run wraps model and embedding calls in a `UsageCollector` so generation calls and embedding calls can be tracked together.

The collector records:

- call kind: `generation` or `embedding`
- model name
- input tokens
- output tokens, when available

`estimate_cost` applies the local pricing table in `app/observability.py` and the agent completion log includes token totals and estimated cost:

```text
agent_completed tool_calls=... tool_used=... latency_ms=... input_tokens=... output_tokens=... cost=...
```

Regular and agent runs also log aggregate request usage:

```text
request_usage input_tokens=... output_tokens=... cost=... model_calls=...
```

When Postgres is available, each completed run is saved to the `traces` table with:

- mode: `rag` or `agent`
- prompt and answer
- model
- tools used
- model call count
- input and output tokens
- estimated cost in USD
- latency
- trajectory metadata

Embedding usage uses the OpenAI embeddings response `prompt_tokens` field and is captured when embeddings are created inside an active usage collection context.

## Run The MCP Server

The project includes a small FastMCP server that exposes the local tools to MCP-compatible clients:

- `search_documents`: searches the indexed local document knowledge base.
- `calculate`: evaluates simple arithmetic expressions.

Start the server with:

```bash
uv run python mcp_server.py
```

The MCP server reuses the same tool implementations as agent mode. For `search_documents`, ingest documents first with `uv run python -m app.ingest` so the local Qdrant collection exists.

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
- [x] Basic MCP server exposing local tools
- [x] Basic token and cost observability
- [x] Postgres trace storage
- [x] Local Docker Compose for Postgres
- [ ] Project polish
