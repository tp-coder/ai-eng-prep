import ast
import operator
import logging

from app.retrieval import retrieve_context


logger = logging.getLogger(__name__)

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


CALCULATOR_TOOL = {
    "type": "function",
    "name": "calculate",
    "description": (
        "Evaluate a simple arithmetic expression. Call this tool for math calculations the user asks for (e.g '2 + 2', '15 * 4.5'). "
        "Return the result of the calculation plus the rational that led to it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expressions using + - * / ** % only. "
                "No function call: no pow(), sqrt() - use ** for powers and ** 0.5 for square roots. "
                "Examples: sqrt(16) = 16 ** 0.5; pow(2, 3) = 2 ** 3."
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    }
}


_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {node}")


def calculator_tool(expression: str) -> str:
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception:
        return (
            f"Could not evaluate expression '{expression}'. Only numbers and + - * / ** and % are supported. "
            "No function calls, no sqrt() or pow() - use ** for powers and ** 0.5 for square roots."
        )


def execute_tool(name: str, args: dict) -> str:
    logger.info("calling tool name=%s args=%s", name, args)
    if name == "search_docs":
        result = search_docs(args["query"])
    elif name == "calculate":
        result = calculator_tool(args["expression"])
    else:
        raise ValueError(f"Unknown tool: {name}")
    logger.info("tool_result name=%s result=%s", name, result[:200])
    return result
