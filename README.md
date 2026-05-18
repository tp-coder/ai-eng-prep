# AI Engineering Prep

This is a prep project to get me ready to dive deeper into AI Engineering and build production_shapep AI Systems.

The project will evolve through 4 stages:

1. LLM API calls and structured output
2. Basic RAG pipeline
3. Tools calling and mini_agent behavior
4. Evals, observability, Docker, and polish

## Setup

### Create a virtual environment and install dependencies

```bash
cd <project_root>
uv sync --extra dev
```

### Copy env file

```bash
cp .env.example .env
```

### Run the app

```bash
uv run python -m app.main
```

### Run tests

```bash
uv run pytest
```

## Current status

- [x] LLM API calls and structured output
- [ ] Basic RAG pipeline
- [ ] Tools calling and mini_agent behavior
- [ ] Evals, observability, Docker, and polish
