"""Parser contract tests. Run with: pytest"""

import pytest

logn_prompts = pytest.importorskip("logn.prompts")


def test_strict_question():
    r = logn_prompts.parse_question("reasoning\nQUESTION: Is it music?")
    assert r.mode == "strict" and r.value == "Is it music?"


def test_loose_question():
    r = logn_prompts.parse_question("I think:\nIs it music?")
    assert r.mode == "loose"


def test_failed_question():
    assert logn_prompts.parse_question("no idea").mode == "failed"


def test_guess_out_of_range_is_not_clamped():
    assert logn_prompts.parse_guess("GUESS: 99", 4).mode == "failed"


def test_answers():
    assert logn_prompts.parse_answer("Yes").value is True
    assert logn_prompts.parse_answer("no.").value is False
    assert logn_prompts.parse_answer("unclear").mode == "failed"


def test_cache_prefix_is_round_invariant():
    docs = [{"title": f"D{i}", "text": "body"} for i in range(4)]
    a = logn_prompts.qbot_prompt(docs, [], 1, 2)
    b = logn_prompts.qbot_prompt(docs, [("q?", True)], 2, 2)
    assert a.cacheable == b.cacheable      # or caching silently stops working
