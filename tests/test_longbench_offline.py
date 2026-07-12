"""Offline (CPU) checks for eval/longbench_eval.py: F1 scorer + truncation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.longbench_eval import (  # noqa: E402
    f1_score, normalize_answer, qa_f1_score, truncate_middle, V1_TASKS,
)


def test_normalize_answer():
    assert normalize_answer("The  Quick, Brown Fox!") == "quick brown fox"
    assert normalize_answer("An apple") == "apple"


def test_f1_exact_match():
    assert f1_score("Paris", "paris") == 1.0
    assert f1_score("The answer is Paris.", "Paris") == 2 * (1 / 3) / (1 + 1 / 3)


def test_f1_no_overlap():
    assert f1_score("London", "Paris") == 0.0


def test_qa_f1_max_over_ground_truths():
    assert qa_f1_score("Paris", ["London", "Paris"]) == 1.0


def test_f1_partial():
    # pred tokens {new, york, city}, gt {new, york}: p=2/3, r=1 -> f1=0.8
    assert abs(f1_score("New York City", "new york") - 0.8) < 1e-9


class FakeTokenizer:
    """Whitespace 'tokenizer' good enough to exercise the truncation logic."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, toks, skip_special_tokens=True):
        return " ".join(toks) + " "


def test_truncate_middle_short_passthrough():
    t = FakeTokenizer()
    assert truncate_middle("a b c", t, 10) == "a b c"


def test_truncate_middle_keeps_head_and_tail():
    t = FakeTokenizer()
    prompt = " ".join(str(i) for i in range(100))
    out = truncate_middle(prompt, t, 10).split()
    assert out == ["0", "1", "2", "3", "4", "95", "96", "97", "98", "99"]


def test_v1_task_table():
    assert set(V1_TASKS) == {"hotpotqa", "2wikimqa", "musique", "multifieldqa_en"}
    for template, max_new in V1_TASKS.values():
        assert "{context}" in template and "{input}" in template
        assert max_new in (32, 64)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
