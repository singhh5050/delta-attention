"""WP-1: variable-stride delta correction and adaptive anchor placement.

The Triton query-sparse kernel (delta_kernel.qsa_kernel) already accepts an
arbitrary sorted index tensor; only the delta *broadcast* in delta_forward
hard-assumes a uniform stride (the reshape to [b, s_p//gamma, gamma, h, d]).
``apply_delta_variable`` generalizes that broadcast: every row i receives the
delta of the nearest anchor <= i.

``plan_adaptive_anchors`` implements the chunked adaptive policy from
docs/WP1_dynamic_stride.md Task B: within a chunk anchors are placed at the
current local stride; after measuring the chunk's mean consecutive-anchor
delta cosine, the stride halves (fast drift = under-sampling), doubles
(smooth), or holds. The sparse pass over the whole sequence is unchanged —
only anchor selection + the dense-row kernel calls + the correction are
chunk-driven. The dense ``cut_n`` tail of delta_forward stays outside the
adaptive region, exactly as in the fixed-stride path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

ADAPT_MARGIN = 0.02  # hysteresis above adapt_threshold before growing gamma


def apply_delta_variable(sparse_out, dense_out_sel, idx, s_p: int):
    """Variable-stride delta correction.

    sparse_out:    [b, s_p, h, d]  sparse attention output rows 0..s_p
    dense_out_sel: [b, n, h, d]    dense rows at the anchor positions ``idx``
    idx:           [n]             ascending anchor row indices, idx[0] == 0
    Returns corrected [b, s_p, h, d]: row i gets sparse_out[i] + delta of the
    nearest anchor <= i. With uniform idx this reproduces the reshape path
    bit-for-bit (same dtype, same add of the same operands) — guarded by T10.
    """
    import torch

    deltas = dense_out_sel - sparse_out[:, idx]
    owner = torch.searchsorted(idx, torch.arange(s_p, device=idx.device), right=True) - 1
    return sparse_out + deltas[:, owner]


def interanchor_cos(deltas) -> Tuple[float, float]:
    """(mean, min) cosine between consecutive anchor deltas. deltas: [b, n, h, d]."""
    import torch

    if deltas.size(1) < 2:
        return 1.0, 1.0
    cos = torch.nn.functional.cosine_similarity(
        deltas[:, :-1].float(), deltas[:, 1:].float(), dim=-1
    )  # [b, n-1, h]
    return cos.mean().item(), cos.min().item()


def next_gamma(cos_mean: float, gamma_c: int, *, threshold: float,
               gamma_min: int, gamma_max: int) -> int:
    """Stride update rule (WP1 Task B): halve on fast drift, double on smooth."""
    if cos_mean < threshold:
        return max(gamma_c // 2, gamma_min)
    if cos_mean > threshold + ADAPT_MARGIN:
        return min(gamma_c * 2, gamma_max)
    return gamma_c


def _sparse_pass_cost(s: int, window: int, sink: int) -> int:
    """Causally-clipped key count of the StreamingLLM pass: row i attends
    min(i+1, window+sink) keys. The uncapped (window+sink)*s estimate goes
    NEGATIVE at short contexts (observed at 4K in WP-1 run 1)."""
    w = min(window + sink, s)
    ramp = w * (w - 1) // 2          # rows 0..w-2 attend i+1 keys
    flat = (s - (w - 1)) * w         # remaining rows attend w keys
    return ramp + flat


def effective_sparsity(anchor_idx, s: int, window: int, sink: int = 1024) -> float:
    """1 - (clipped window cost + sum of anchor row costs) / (s^2 / 2).

    Anchor row i attends to i+1 keys (causal dense row). Comparable across
    fixed and adaptive configs; the dense tail rows are included by the
    caller via ``anchor_idx`` covering them.
    """
    dense_cost = int(anchor_idx.sum().item()) + anchor_idx.numel()  # sum(i+1)
    full = s * s / 2.0
    return 1.0 - (_sparse_pass_cost(s, window, sink) + dense_cost) / full


def plan_chunks(s_p: int, chunk: int) -> List[Tuple[int, int]]:
    return [(a, min(a + chunk, s_p)) for a in range(0, s_p, chunk)]


def uniform_effective_sparsity(s: int, gamma: int, window: int, sink: int = 1024) -> float:
    """Pure-python effective_sparsity for the fixed-stride path (no torch).

    Mirrors delta_forward's fixed layout: anchors at 0, gamma, ... < s_p plus
    the dense cut_n tail rows s_p..s-1, where cut_n = s % gamma + max(128, gamma).
    Reported for fixed-γ configs so adaptive/fixed frontiers are comparable.
    """
    cut_n = s % gamma + max(128, gamma)
    s_p = s - cut_n
    n = (s_p + gamma - 1) // gamma
    anchor_cost = gamma * n * (n - 1) // 2 + n              # sum(i+1) over anchors
    tail_cost = (s * (s + 1) - s_p * (s_p + 1)) // 2        # sum(i+1) over tail rows
    full = s * s / 2.0
    return 1.0 - (_sparse_pass_cost(s, window, sink) + anchor_cost + tail_cost) / full
