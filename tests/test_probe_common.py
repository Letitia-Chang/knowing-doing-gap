import pytest

import common
from common import ACTION_SYSTEMS, action_user_prompt, necessity_labels, parse_tool_decision, surface_features, threshold_column
from run_action import BASELINE_SYSTEM, INTERVENTION_SYSTEM, choose_action

Q = {"id": "Q0001", "category": "arithmetic", "question": "What is 12 * 34 + 5?"}


class FakeClient:
    def __init__(self):
        self.calls = []

    def ask(self, prompt, system=None, temperature=0.0, max_tokens=None):
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return '{"action":"answer","answer":"413"}'


@pytest.mark.parametrize("intervention", [False, True])
def test_prompts_identical_to_run_action(intervention):
    """The probe study must see exactly the decision context run_action.py gives the model."""
    client = FakeClient()
    choose_action(client, Q, intervention)
    sent = client.calls[0]
    assert sent["prompt"] == action_user_prompt(Q)
    assert sent["system"] == ACTION_SYSTEMS["intervention" if intervention else "baseline"]
    assert sent["temperature"] == 0.0  # greedy, as in the local action stage


def test_action_systems_are_the_original_constants():
    assert ACTION_SYSTEMS["baseline"] is BASELINE_SYSTEM and ACTION_SYSTEMS["intervention"] is INTERVENTION_SYSTEM


def test_necessity_labels_thresholds():
    all_right = necessity_labels(10, 10)
    assert not any(all_right[threshold_column(t)] for t in (1.0, 0.8, 0.5))
    one_wrong = necessity_labels(9, 10)
    assert one_wrong["tool_needed"] and not one_wrong["tool_needed_acc80"] and not one_wrong["tool_needed_acc50"]
    mostly_wrong = necessity_labels(7, 10)
    assert mostly_wrong["tool_needed"] and mostly_wrong["tool_needed_acc80"] and not mostly_wrong["tool_needed_acc50"]
    assert necessity_labels(4, 10)["tool_needed_acc50"]
    assert necessity_labels(3, 10)["necessity_accuracy"] == pytest.approx(0.3)


def test_threshold_column_names():
    assert threshold_column(1.0) == "tool_needed"
    assert threshold_column(0.8) == "tool_needed_acc80"
    assert threshold_column(0.5) == "tool_needed_acc50"


@pytest.mark.parametrize(
    "obj,raw,called,tool,malformed",
    [
        ({"action": "tool", "tool": "calculator", "arguments": {"expression": "1+1"}}, "", True, "calculator", False),
        ({"action": "search", "arguments": {"query": "x"}}, "", True, "search", False),
        ({"action": "calculator", "arguments": {}}, "", True, "calculator", False),
        ({"action": "answer", "answer": "4"}, "", False, None, False),
        (63, "63", False, None, False),
        ("Paris", '"Paris"', False, None, False),
        ({"action": "think"}, "raw", False, None, True),
        (None, "not json", False, None, True),
    ],
)
def test_parse_tool_decision_mirrors_run_action(obj, raw, called, tool, malformed):
    d = parse_tool_decision(obj, raw)
    assert (d["tool_called"], d["tool"], d["malformed"]) == (called, tool, malformed)


def test_parse_tool_decision_direct_answer_values():
    assert parse_tool_decision({"action": "answer", "answer": "4"}, "")["direct_answer"] == "4"
    assert parse_tool_decision(63, "63")["direct_answer"] == "63"


def test_surface_features():
    f = surface_features(Q)
    assert f["n_digits"] == 5 and f["n_numbers"] == 3 and f["max_number_len"] == 2 and f["is_arithmetic"] == 1.0
    assert f["n_operators"] == 2
    assert surface_features({"category": "factual", "question": "Capital of France?"})["is_arithmetic"] == 0.0
