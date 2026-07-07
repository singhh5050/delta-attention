"""WP-2 Task A: differentiable delta forward (docs/WP2_trainable_delta.md).

The inference path cannot backprop: qsa_kernel is a forward-only Triton
kernel and the sparse side uses hip-attn inference kernels. This module
rebuilds the SAME math with autograd-capable ops:

- sparse side: torch FlexAttention with a block mask = causal AND
  (col < sink OR row - col < window)  — the StreamingLLM pattern.
- query-sparse side: the anchor rows (uniform gamma, plus the dense cut_n
  tail replicating delta_forward's layout) via plain SDPA with a
  row-restricted causal mask — anchor count is small.
- delta = dense_anchor - sparse[anchors], broadcast within each gamma block.

Everything stays in the graph; ``detach_delta=True`` exists ONLY as the
ablation arm (gradient flows through the sparse path but not the reused
correction).

Equivalence caveat (T13): hip-attn's window path applies the sliding window
at *block* granularity (block_size_q/k), so a handful of extra keys near the
window edge are attended relative to this exact mask. If T13 fails, the fix
is to reconcile THIS mask_mod to hip's block-granular semantics — never to
loosen the tolerance.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_flex = None


def _get_flex():
    """Compile flex_attention once (required for reasonable performance)."""
    global _flex
    if _flex is None:
        _flex = torch.compile(flex_attention, dynamic=False)
    return _flex


@functools.lru_cache(maxsize=8)
def get_block_mask(s: int, window: int, sink: int, device: str):
    """Block mask for causal StreamingLLM sparsity; cached per (s, window, sink)."""

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((kv_idx < sink) | (q_idx - kv_idx < window))

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=s, KV_LEN=s, device=device)


def anchor_layout(s: int, gamma: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Anchor rows + dense tail rows, replicating delta_forward's cut_n layout."""
    cut_n = s % gamma + max(128, gamma)
    s_p = s - cut_n
    return torch.arange(0, s_p, gamma), torch.arange(s_p, s), s_p


def delta_forward_train(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: int,
    window: int,
    sink: int = 1024,
    scaling: Optional[float] = None,
    detach_delta: bool = False,
) -> torch.Tensor:
    """Differentiable delta-corrected attention.

    q: [b, h, s, d]; k/v: [b, h_kv, s, d] (GQA repeated internally, in-graph).
    Returns [b, s, h, d] to match inference delta_forward's output layout.
    """
    b, h, s, d = q.shape
    if k.size(1) != h:
        rep = h // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    if scaling is None:
        scaling = d ** -0.5

    bm = get_block_mask(s, int(window), int(sink), str(q.device))
    sparse = _get_flex()(q, k, v, block_mask=bm, scale=scaling)
    sparse = sparse.transpose(1, 2)  # [b, s, h, d]

    idx, tail, s_p = anchor_layout(s, gamma)
    idx, tail = idx.to(q.device), tail.to(q.device)
    sel = torch.cat((idx, tail))

    key_pos = torch.arange(s, device=q.device)
    row_mask = (key_pos.unsqueeze(0) <= sel.unsqueeze(1)).view(1, 1, sel.numel(), s)
    dense_sel = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, sel], k, v, attn_mask=row_mask, scale=scaling
    ).transpose(1, 2)  # [b, n_sel, h, d]
    dense_anchor = dense_sel[:, : idx.numel()]
    dense_tail = dense_sel[:, idx.numel():]

    delta = dense_anchor - sparse[:, idx]
    if detach_delta:  # ablation arm ONLY
        delta = delta.detach()

    n = idx.numel()
    corrected = (
        sparse[:, :s_p].reshape(b, n, gamma, h, d) + delta.reshape(b, n, 1, h, d)
    ).reshape(b, s_p, h, d)
    return torch.cat((corrected, dense_tail), dim=1)


def anchor_grad_ratio(q_grad: torch.Tensor, gamma: int) -> float:
    """Mandatory training metric: mean per-row grad norm at anchor rows vs
    non-anchor rows (docs/WP2 T14 — expect concentration at anchors)."""
    s = q_grad.size(2)
    idx, tail, s_p = anchor_layout(s, gamma)
    device = q_grad.device
    norms = q_grad.float().norm(dim=-1).mean(dim=1)[0]  # [s] mean over heads, b=1
    is_dense = torch.zeros(s, dtype=torch.bool, device=device)
    is_dense[idx.to(device)] = True
    is_dense[tail.to(device)] = True
    return (norms[is_dense].mean() / norms[~is_dense].mean().clamp_min(1e-12)).item()
