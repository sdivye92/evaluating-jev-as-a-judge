import random

from jevjudge.data import clean_helpsteer_prompt, wrong_number
from jevjudge.mbpp_exec import make_incorrect, mutate, normalize, run_tests

GOLD = """
def count_positive(xs):
    # count items above zero
    n = 0
    for x in xs:
        if x > 0:
            n = n + 1
    return n
"""
TESTS = [
    "assert count_positive([1, -2, 3]) == 2",
    "assert count_positive([]) == 0",
    "assert count_positive([0, 0, 5]) == 1",
]


def test_run_tests_pass_fail_timeout():
    assert run_tests(GOLD, [], TESTS).passed
    bad = run_tests(GOLD.replace("x > 0", "x >= 0"), [], TESTS)
    assert not bad.passed and bad.assertion_failed
    loop = run_tests("def count_positive(xs):\n    while True:\n        pass", [], TESTS)
    assert not loop.passed and not loop.assertion_failed


def test_normalize_strips_comments_so_gold_and_mutant_look_alike():
    assert "#" not in normalize(GOLD)
    assert normalize(GOLD) == normalize(normalize(GOLD))


def test_mutants_run_but_fail_an_assertion():
    gold = normalize(GOLD)
    made = [make_incorrect(gold, [], TESTS, random.Random(s)) for s in range(10)]
    made = [m for m in made if m]
    assert made, "expected at least one failing mutant"
    for code, res in made:
        assert code != gold and res.assertion_failed and not res.passed


def test_mutate_changes_code_or_reports_no_site():
    assert mutate("def f():\n    return 'x'", random.Random(0)) is None
    assert mutate(normalize(GOLD), random.Random(0)) != normalize(GOLD)


def test_wrong_number_is_plausible_and_distinct():
    rng = random.Random(0)
    for gold in (7, 20, 144, 1000):
        for _ in range(50):
            w = wrong_number(gold, "He had 12 apples and bought 5 more, then ate 3.", rng)
            assert w != gold and w >= 0


def test_helpsteer_prompt_cleaning():
    raw = "What is 2+2?<extra_id_1>Assistant\n4<extra_id_1>User\nAnd 3+3?"
    out = clean_helpsteer_prompt(raw)
    assert "<extra_id" not in out and "Assistant: 4" in out and "User: And 3+3?" in out
