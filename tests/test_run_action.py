from run_action import extract_json


def test_extract_json_clean_object():
    assert extract_json('{"action":"answer","answer":"4"}') == {"action": "answer", "answer": "4"}


def test_extract_json_object_embedded_in_prose():
    text = 'Sure, here you go:\n{"action":"tool","tool":"calculator","arguments":{"expression":"2+2"}}\nHope that helps.'
    parsed = extract_json(text)
    assert parsed == {
        "action": "tool",
        "tool": "calculator",
        "arguments": {"expression": "2+2"},
    }


def test_extract_json_bare_number_is_not_a_dict():
    # Some models return a bare JSON primitive instead of the expected object;
    # run_action.py treats this as a direct answer rather than crashing.
    assert extract_json("63") == 63


def test_extract_json_bare_string():
    assert extract_json('"Paris"') == "Paris"


def test_extract_json_unparsable_returns_none():
    assert extract_json("I am not JSON at all.") is None
