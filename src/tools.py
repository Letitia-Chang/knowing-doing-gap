from __future__ import annotations

import ast
import operator as op
from pathlib import Path
from typing import Any

from io_utils import read_json

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_ast(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError(f"Unsupported calculator expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    expression = expression.replace("×", "*").replace("÷", "/")
    tree = ast.parse(expression, mode="eval")
    result = _eval_ast(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def mock_search(query: str, question_id: str | None = None, mock_path: str | Path = "data/mock_search.json") -> str:
    data = read_json(mock_path)
    if question_id and question_id in data:
        return data[question_id]
    q = query.lower()
    for key, value in data.items():
        if key.lower() in q:
            return value
    # fallback: crude keyword matching
    for value in data.values():
        words = [w for w in q.split() if len(w) > 4]
        if any(w in value.lower() for w in words):
            return value
    return "Search result: No matching mock result found. Try a more specific query."


def execute_tool(tool: str, arguments: dict[str, Any], question_id: str | None = None, mock_path: str | Path = "data/mock_search.json") -> str:
    if tool == "calculator":
        return calculator(str(arguments.get("expression", "")))
    if tool == "search":
        return mock_search(str(arguments.get("query", "")), question_id=question_id, mock_path=mock_path)
    raise ValueError(f"Unknown tool: {tool}")
