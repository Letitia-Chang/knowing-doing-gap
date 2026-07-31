from pathlib import Path

import pytest

from tools import calculator, execute_tool, mock_search

MOCK_SEARCH_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_search.json"


def test_calculator_basic_arithmetic():
    assert calculator("2 + 2") == "4"


def test_calculator_order_of_operations():
    assert calculator("94 + 40 * 50 - 24 * 87") == "6"


def test_calculator_rejects_unsafe_expression():
    # The calculator only walks a whitelisted AST, so arbitrary code
    # (attribute access, function calls, etc.) must be rejected, not executed.
    with pytest.raises(Exception):
        calculator("__import__('os').system('echo pwned')")


def test_mock_search_by_question_id():
    result = mock_search("anything", question_id="F01", mock_path=MOCK_SEARCH_PATH)
    assert "Paris" in result


def test_mock_search_unknown_query_has_fallback():
    result = mock_search("completely unrelated gibberish query", mock_path=MOCK_SEARCH_PATH)
    assert "No matching mock result" in result


def test_execute_tool_dispatches_calculator():
    assert execute_tool("calculator", {"expression": "3 * 4"}) == "12"


def test_execute_tool_rejects_unknown_tool():
    with pytest.raises(ValueError):
        execute_tool("not_a_real_tool", {})
