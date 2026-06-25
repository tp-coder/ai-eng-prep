# ai-eng-prep — Roadmap

Hands-on AI-engineering build: a production-shaped **agent + RAG + eval** system that doubles as portfolio + interview proof.

**Strategy — build each concept twice:** Python here (depth + the credential repo), TS/Node in the Aircall mirror (native-stack interview readiness). Understand-once, build-twice; the TS build *is* the live-coding rehearsal.
Companion: `~/workspace/scratch/08.tp-jobs/aircall/aircall-drill-roadmap.md`

---

## Phase 1 — Tool-calling agent ✅ (2026-06-25)
Retrieval flipped from always-on → **model-decided**.
- `retrieval.py` — `retrieve_context()` + helpers (one source of truth)
- `tools.py` — `SEARCH_DOCS_TOOL` (flat Responses-API schema) + `search_docs` / `execute_tool`
- `llm.py` — `complete_with_tools()` agent loop: Responses API + `tools=` + `text_format`; appends `output` back + `function_call_output` w/ matching `call_id`; **MAX_TOOL_ITERATIONS=5** cap; `tool_calls=N` logging
- `main.py` — `--agent` flag
- ✅ Falsifiable test: doc-Q → `tool_calls=1` (grounded + sources); general-Q → `tool_calls=0`

## Phase 2 — Second tool + selection  ⬅️ NEXT
Add a distinct tool (e.g. `calculator`) so the agent must pick the **right** tool, not just tool-or-nothing. The loop already supports N tools (just add def + executor branch + register) — extensibility payoff.

## Phase 3 — Trajectory / agent eval  *(the moat)*
Extend the eval harness from retrieval-only to **agent behavior**: right tool? grounded when it should be? no needless calls? Regression-gated. → "an agent *with an eval suite that proves it picks the right tool*."

## Phase 4 — MCP server  *(the headline)*
Expose the tools via an **MCP server**; consume from an external client (Claude Desktop). → "I built an MCP." Closes the loop with the EIS / Aircall / RHG MCP conversations.

## Phase 5 — Dual-mode + write-up
`main.py` offers both modes (always-on RAG vs agentic) for comparison; then an article + project case study on tp-coder.dev.
