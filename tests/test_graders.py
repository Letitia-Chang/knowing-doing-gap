from graders import extract_last_number, grade_answer, grade_arithmetic, grade_factual


def test_extract_last_number_plain():
    assert extract_last_number("The answer is 63.") == 63


def test_extract_last_number_ignores_base_annotation():
    # Regression test: "63 in base 10" was previously mis-graded because the
    # naive last-number match picked up the trailing "10" instead of "63".
    assert extract_last_number("63 in base 10") == 63


def test_extract_last_number_none_when_no_digits():
    assert extract_last_number("no numbers here") is None


def test_grade_arithmetic_matches_reference():
    assert grade_arithmetic("9 times 7 is 63 in base 10", "63") is True


def test_grade_arithmetic_rejects_wrong_answer():
    assert grade_arithmetic("The answer is 64.", "63") is False


def test_grade_factual_accepts_alternate_phrasing():
    assert grade_factual("It's William Shakespeare.", "Shakespeare", accept=["William Shakespeare"]) is True


def test_grade_factual_case_and_punctuation_insensitive():
    assert grade_factual("PARIS!", "Paris") is True


def test_grade_answer_dispatches_by_category():
    arithmetic_q = {"category": "arithmetic", "reference_answer": "4"}
    factual_q = {"category": "factual", "reference_answer": "Paris"}
    assert grade_answer(arithmetic_q, "2 + 2 = 4") is True
    assert grade_answer(factual_q, "The capital of France is Paris.") is True
