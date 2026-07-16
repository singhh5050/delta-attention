"""Offline (CPU) checks for eval/specdec_eval.py's acceptance rule.

The sequential-acceptance invariant is THE correctness core of speculative
decoding: every emitted token must be the dense model's greedy choice given
a dense-grade prefix. accept_block is pure, so exactness is testable here."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.specdec_eval import (  # noqa: E402
    DRAFT_WEIGHTS, accept_block, classify_parity, positional_stats,
)


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


def test_positional_stats_curve_and_genuine():
    # blocks of K=4 with accepted-prefix lengths 4, 2, 0:
    # pos_acc[i] = frac(nacc > i) -> [2/3, 2/3, 1/3, 1/3]
    pos_acc, genuine = positional_stats([4, 2, 0], 4)
    assert pos_acc == [2 / 3, 2 / 3, 1 / 3, 1 / 3]
    # genuine = accepted positions 2..K only: (3 + 1 + 0) / (3 blocks * 3)
    assert abs(genuine - 4 / 9) < 1e-12
    # headline == genuine reconstruction: acc = (pos1 + genuine*(K-1)) / K
    acc = sum([4, 2, 0]) / (3 * 4)
    assert abs(acc - (pos_acc[0] + genuine * 3) / 4) < 1e-12


def test_positional_stats_edge_cases():
    # all-full blocks: every position 1.0, genuine 1.0
    pos_acc, genuine = positional_stats([2, 2], 2)
    assert pos_acc == [1.0, 1.0] and genuine == 1.0
    # K=1 has no drafted positions beyond the anchor; genuine == pos1
    pos_acc, genuine = positional_stats([1, 0], 1)
    assert pos_acc == [0.5] and genuine == 0.5
    # empty (no counted blocks) must not divide by zero
    pos_acc, genuine = positional_stats([], 4)
    assert pos_acc == [0.0] * 4 and genuine == 0.0


def test_classify_parity_gate():
    FULL = 10**9
    # all byte-identical -> "full", no failure
    assert classify_parity([FULL, FULL], 24) == ("full", None)
    # late divergence (>= gate) passes with the numeric min
    val, fail = classify_parity([75, FULL], 24)
    assert val == 75 and fail is None
    # early NON-TIE divergence hard-fails
    val, fail = classify_parity([17, FULL], 24)
    assert val == 17 and fail is not None
    # early divergence proven to be a bf16 tie is benign and annotated
    val, fail = classify_parity([("tie", 17, 0.125), FULL], 24)
    assert fail is None and val.startswith("full+1tie(17@0.12")
    # a tie does NOT excuse a separate early non-tie divergence
    val, fail = classify_parity([("tie", 17, 0.0), 5], 24)
    assert fail is not None
    # length mismatch after a clean prefix (-1) is always a bug
    _, fail = classify_parity([-1, FULL], 24)
    assert fail is not None
    # every prompt tie-flipped: no certified prefix, but not a failure
    val, fail = classify_parity([("tie", 30, 0.25)], 24)
    assert fail is None and val.startswith("tie-only")
    # no checks -> (None, None), and gate disabled (0) never fails
    assert classify_parity([], 24) == (None, None)
    assert classify_parity([3], 0) == (3, None)


def test_draft_weights_registry():
    assert DRAFT_WEIGHTS["base"] == ""
    assert set(DRAFT_WEIGHTS) == {"base", "ce32k", "dft", "dftmix"}


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
