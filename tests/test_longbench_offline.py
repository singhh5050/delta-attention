"""Offline (CPU) checks for eval/longbench_eval.py: F1 scorer + truncation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.longbench_eval import (  # noqa: E402
    DECODE_ARMS, ENMC_TEMPLATE, enmc_correct_letter, f1_score,
    FIRST_LINE_ONLY, GOVREPORT_TEMPLATE, lb_official, NO_CHAT_WRAP,
    normalize_answer, qa_f1_score, truncate_middle, V1_FULL_EN, V1_TASKS,
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


def test_enmc_template_placeholders():
    for field in ("{context}", "{question}", "{A}", "{B}", "{C}", "{D}"):
        assert field in ENMC_TEMPLATE


def test_enmc_correct_letter():
    ex = {"options": ["red", "green", "blue", "yellow"], "answer": ["green"]}
    assert enmc_correct_letter(ex) == "B"
    # whitespace on either side must not break the match
    ex = {"options": ["red ", "green", "blue", "yellow"], "answer": [" red"]}
    assert enmc_correct_letter(ex) == "A"


def test_enmc_correct_letter_unmatched_is_none():
    assert enmc_correct_letter(
        {"options": ["a", "b", "c", "d"], "answer": ["e"]}) is None
    assert enmc_correct_letter(
        {"options": ["a", "b", "c", "d"], "answer": []}) is None


def test_govreport_template():
    assert "{context}" in GOVREPORT_TEMPLATE
    # rendered via .replace, so no other format fields may exist
    assert GOVREPORT_TEMPLATE.count("{") == GOVREPORT_TEMPLATE.count("{context}")


def test_v1full_vendored_files_cover_all_tasks():
    """The vendored THUDM/LongBench files (templates/maxlens/metrics) must
    resolve every English task; requires jieba/fuzzywuzzy/rouge (chain setup
    installs them before this gate runs)."""
    prompts, maxlens, metrics = lb_official()
    assert len(V1_FULL_EN) == 16
    for t in V1_FULL_EN:
        assert "{context}" in prompts[t], t
        assert isinstance(maxlens[t], int) and maxlens[t] > 0, t
        assert callable(metrics[t]), t
    # protocol sets must only name real tasks (lsht is zh-only: in the
    # official lists but never run by v1full)
    assert NO_CHAT_WRAP - {"lsht"} <= set(V1_FULL_EN)
    assert FIRST_LINE_ONLY - {"lsht"} <= set(V1_FULL_EN)
    # str.format only parses the template, so brace-laden code contexts pass
    assert "{x: 1}" in prompts["lcc"].format(context="return {x: 1}")


def test_v1full_metric_families():
    """One canary per metric family, scored through the OFFICIAL functions."""
    _, _, metrics = lb_official()
    assert metrics["passage_count"]("The final answer is: 7", "7") == 1.0
    assert metrics["passage_retrieval_en"]("Paragraph 12", "Paragraph 12") == 1.0
    assert metrics["trec"]("location", "location",
                           all_classes=["location", "human"]) == 1.0
    assert metrics["lcc"]("return x + 1", "return x + 1") == 1.0
    assert metrics["gov_report"]("a summary", "a summary") > 0.99
    assert metrics["hotpotqa"]("Paris", "Paris", all_classes=None) == 1.0


def test_decode_arm_gamma_sweep():
    assert DECODE_ARMS["sparse_dec"] == ("sparse", None)
    for g in (2, 4, 8, 16):
        assert DECODE_ARMS[f"delta_dec{g}"] == ("delta", g)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
