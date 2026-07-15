"""Offline (CPU) checks for eval/specdec_eval.py's acceptance rule.

The sequential-acceptance invariant is THE correctness core of speculative
decoding: every emitted token must be the dense model's greedy choice given
a dense-grade prefix. accept_block is pure, so exactness is testable here."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.specdec_eval import DRAFT_WEIGHTS, accept_block  # noqa: E402


def test_full_block_accepted_returns_bonus():
    n, nxt, full = accept_block([5, 6, 7], [5, 6, 7], bonus=9)
    assert (n, nxt, full) == (3, 9, True)


def test_first_mismatch_takes_dense_token():
    # proposal 6 != dense 8 at position 1 -> accept 1, emit dense's 8
    n, nxt, full = accept_block([5, 6, 7], [5, 8, 7], bonus=9)
    assert (n, nxt, full) == (1, 8, False)


def test_zero_accepted():
    n, nxt, full = accept_block([5, 6], [4, 6], bonus=9)
    assert (n, nxt, full) == (0, 4, False)
    # later agreement after an earlier mismatch must NOT resurrect tokens
    n, nxt, full = accept_block([5, 6, 7], [4, 6, 7], bonus=9)
    assert (n, nxt, full) == (0, 4, False)


def test_single_token_block():
    assert accept_block([3], [3], bonus=4) == (1, 4, True)
    assert accept_block([3], [2], bonus=4) == (0, 2, False)


def test_emitted_tokens_always_dense_choices():
    # invariant: proposals[:n_accepted] == dense_choices[:n_accepted] and the
    # substitute token is dense's — so every emitted token is dense-greedy
    cases = [([1, 2, 3], [1, 2, 3], 7), ([1, 2, 3], [1, 9, 3], 7),
             ([1], [2], 7), ([4, 4], [4, 4], 0)]
    for props, dense, bonus in cases:
        n, nxt, full = accept_block(props, dense, bonus)
        assert props[:n] == dense[:n]
        assert nxt == (bonus if full else dense[n])


def test_draft_weights_registry():
    assert DRAFT_WEIGHTS["base"] == ""
    assert set(DRAFT_WEIGHTS) == {"base", "ce32k", "dft"}


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
