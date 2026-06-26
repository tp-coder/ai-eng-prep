from mcp.server.fastmcp import FastMCP
from app.tools import search_docs as _search_docs, calculator_tool as _calculator_tool

mcp = FastMCP("ai-engineering-prep")


@mcp.tool()
def search_documents(query: str) -> str:
    """
    Search the local document knowledge base for passages relevant to a question.
    Use when the answer might live in the indexed documents.
    Return the most relevant passages from the knowledge base, or a note that nothing relevant was found.
    """
    return _search_docs(query)


@mcp.tool()
def calculate(expression: str) -> str:
    """
    Evaluate a simple arithmetic expression using numbers and + - * / ** % only.
    No function calls, no sqrt() or pow() - use ** for powers and ** 0.5 for square roots.
    """
    return _calculator_tool(expression)


if __name__ == "__main__":
    mcp.run()
