"""Offline (no torch/GPU) tests for WP-1 stride planning math."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delta_attention.stride import (ADAPT_MARGIN, next_gamma,  # noqa: E402
                                    plan_chunks, uniform_effective_sparsity)


def test_next_gamma_rules():
    kw = dict(threshold=0.95, gamma_min=16, gamma_max=256)
    assert next_gamma(0.90, 64, **kw) == 32          # fast drift -> halve
    assert next_gamma(0.99, 64, **kw) == 128         # smooth -> double
    assert next_gamma(0.96, 64, **kw) == 64          # inside hysteresis band -> hold
    assert next_gamma(0.95 + ADAPT_MARGIN, 64, **kw) == 64  # boundary is hold
    assert next_gamma(0.0, 16, **kw) == 16           # clamped at gamma_min
    assert next_gamma(1.0, 256, **kw) == 256         # clamped at gamma_max


def test_plan_chunks():
    assert plan_chunks(8192, 4096) == [(0, 4096), (4096, 8192)]
    assert plan_chunks(5000, 4096) == [(0, 4096), (4096, 5000)]
    assert plan_chunks(100, 4096) == [(0, 100)]


def test_uniform_effective_sparsity():
    # denser anchors (smaller gamma) => less sparse
    s, w = 131072, 2048
    es = {g: uniform_effective_sparsity(s, g, w) for g in (16, 32, 64, 128, 256)}
    assert all(0.0 < v < 1.0 for v in es.values())
    assert es[16] < es[32] < es[64] < es[128] < es[256]
    # WP1's formula also charges the window pass (unlike the paper's 98.5%
    # queries-only figure): at 131K the window costs ~4.7%, anchors ~1.6%.
    assert 0.90 < es[64] < 0.96
    # at 4K with a 2K window+sinks, sparse+anchors costs ~as much as dense —
    # near-zero (can dip slightly negative); matches why 4K can't
    # discriminate gammas (PoC run 1 observation)
    assert -0.1 < uniform_effective_sparsity(4096, 64, 2048) < 0.1
