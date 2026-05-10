# AI Engineering Prep

This is a prep project to get me ready to dive deeper into AI Engineering and build production_shapep AI Systems.

The project will evolve through 4 stages:

1. LLM API calls and structured output
2. Basic RAG pipeline
3. Tools calling and mini_agent behavior
4. Evals, observability, Docker, and polish

## Setup

### Create a virtual environment

```bash
cd <project_root>
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -e ".[dev]"
```

### Copy env file

```bash
cp .env.example .env
```

### Run the app

```bash
python3 -m app.main
```

### Run tests

```bash
pytest
```

## Current status

Project skeleton. No LLM calls yet