from app.retrieval import retrieve_context

SEARCH_DOCS_TOOL = {
    "type": "function",
    "name": "search_docs",
    "description": (
        "Search the local document knowledge base for passages relevant to a question. "
        "Call this when answering need facts that might be in the indexed documents. "
        "Return the most relevant passages from the knowledge base, or a note that nothing relevant was found."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused question or keywords to retrieve relevant passages"
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def search_docs(query: str) -> str:
    context = retrieve_context(query)
    return context or "No indexed passages were relevant to this question."


def execute_tool(name: str, args: dict) -> str:
    if name == "search_docs":
        return search_docs(args["query"])
    raise ValueError(f"Unknown tool: {name}")
