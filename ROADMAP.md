# ai-eng-prep — Roadmap

Hands-on AI-engineering build: a production-shaped **agent + RAG + eval** system that doubles as portfolio + interview proof.

**Strategy — build each concept twice:** Python here (depth + the credential repo), TS/Node in the Aircall mirror (native-stack interview readiness). Understand-once, build-twice; the TS build _is_ the live-coding rehearsal.
Companion: `~/workspace/scratch/08.tp-jobs/aircall/aircall-drill-roadmap.md`

---

## Phase 1 — Tool-calling agent ✅ (2026-06-25)

Retrieval flipped from always-on → **model-decided**.

- `retrieval.py` — `retrieve_context()` + helpers (one source of truth)
- `tools.py` — `SEARCH_DOCS_TOOL` (flat Responses-API schema) + `search_docs` / `execute_tool`
- `llm.py` — `complete_with_tools()` agent loop: Responses API + `tools=` + `text_format`; appends `output` back + `function_call_output` w/ matching `call_id`; **MAX_TOOL_ITERATIONS=5** cap; `tool_calls=N` logging
- `main.py` — `--agent` flag
- ✅ Falsifiable test: doc-Q → `tool_calls=1` (grounded + sources); general-Q → `tool_calls=0`

## Phase 2 — Second tool + selection ✅ (2026-06-25)

Add a distinct tool (e.g. `calculator`) so the agent must pick the **right** tool, not just tool-or-nothing. The loop already supports N tools (just add def + executor branch + register) — extensibility payoff.

## Phase 3 — Trajectory / agent eval ✅ (2026-06-25) _(the moat)_

Eval harness extended from retrieval-only to **agent behavior**.

- `complete_with_tools` now returns `tool_names` (the trajectory)
- `evals/agent_dataset.json` — cases: math / doc / general / compound, each with `expected_tools`
- `evals/run_agent.py` — **set-equality** on tools called + **precision/recall/accuracy** metrics, pass/fail table, regression gate
- ✅ 4/4 pass · Accuracy 1.0 · Tool precision 1.0 · Tool recall 1.0
- → "an agent _with an eval suite that proves it picks the right tool_ — measured with precision & recall."

## Phase 4 — MCP server ✅ (2026-06-26) _(the headline)_

Exposed the existing tools over **MCP** — reusing the same `search_docs` / `calculate` executors (capability vs. protocol).

- `mcp_server.py` — `FastMCP("ai-engineering-prep")`, `@mcp.tool()` wrapping the existing executors; docstrings carry the Phase-2 syntax lesson into the tool schema
- ✅ Verified in the **MCP Inspector** (`uv run mcp dev`) — both tools return
- ✅ Added to **Claude Code** (`claude mcp add`) and **consumed live by a Claude Code agent** (search_documents → grounded notes; calculate → 1024) — the LLM client does the NL→argument extraction the non-LLM Inspector couldn't
- → "I built an MCP server, and my own Claude Code agent consumes it." Closes the loop with the EIS / Aircall / RHG MCP conversations.

## Phase 5 — Dual-mode + write-up

`main.py` offers both modes (always-on RAG vs agentic) for comparison; then an article + project case study on tp-coder.dev.
