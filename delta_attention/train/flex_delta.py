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

Mask semantics (reconciled to hip-attn 1.2.9, attention_extend_bsa.py,
BLOCKWISE_MASKING=1 default): within each query block of Q_BLOCK rows the
window is anchored at the block's LAST row (`seq_len = max(pos_tdst)`), so a
key is attended iff  causal AND (kv < sink OR kv + window > block_last_row).
Earlier rows in a block therefore see up to Q_BLOCK-1 fewer of their oldest
window keys than a per-row window would give. This was verified against the
kernel source after T13 initially failed at mean cos 0.9938 with a per-row
window mask. If T13 regresses again: reconcile THIS mask, not the tolerance.
"""

from __future__ import annotations

import functools
import os
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


Q_BLOCK = 64  # hip's block_sparse_block_size_q (config.get_hip_config, last stage)


@functools.lru_cache(maxsize=8)
def get_block_mask(s: int, window: int, sink: int, device: str):
    """Causal StreamingLLM mask with hip's block-anchored window; cached.

    Window anchored at the last row of each Q_BLOCK query block, matching
    hip-attn's BLOCKWISE_MASKING kernel semantics (see module docstring).
    """

    def mask_mod(b, h, q_idx, kv_idx):
        block_last = (q_idx // Q_BLOCK) * Q_BLOCK + (Q_BLOCK - 1)
        return (q_idx >= kv_idx) & ((kv_idx < sink) | (kv_idx + window > block_last))

    # _compile=True builds the BlockMask directly from mask_mod; the eager
    # path first materializes the full s x s boolean mask, whose block-sum
    # intermediate is O(s^2) — 128GB at 131K (OOM'd the anchorbench long
    # ladder, 07-22). Same BlockMask either way; lru_cache pays the compile
    # once per shape.
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=s, KV_LEN=s,
                             device=device, _compile=True)


def anchor_layout(s: int, gamma: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Anchor rows + dense tail rows, replicating delta_forward's cut_n layout."""
    cut_n = s % gamma + max(128, gamma)
    s_p = s - cut_n
    return torch.arange(0, s_p, gamma), torch.arange(s_p, s), s_p


class _ScaleGrad(torch.autograd.Function):
    """Identity forward; backward multiplies the gradient by a constant.
    Applied to the correction term ONLY — Jeff's 1/γ intervention: damp the
    γ-times-summed gradient through the broadcast delta without touching the
    forward math or the sparse path's gradient."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def delta_forward_train(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: int,
    window: int,
    sink: int = 1024,
    scaling: Optional[float] = None,
    detach_delta: bool = False,
    delta_grad_scale: float = 1.0,
) -> torch.Tensor:
    """Differentiable delta-corrected attention.

    q: [b, h, s, d]; k/v: [b, h_kv, s, d] (GQA repeated internally, in-graph).
    Returns [b, s, h, d] to match inference delta_forward's output layout.
    """
    b, h, s, d = q.shape
    assert s % Q_BLOCK == 0, (
        f"s={s} must be a multiple of {Q_BLOCK} (hip block-anchored window "
        "semantics are only reconciled for full query blocks)")
    if scaling is None:
        scaling = d ** -0.5

    # DELTA_SPARSE_IMPL: sparse-branch kernel selector — a BENCH DIAGNOSTIC
    # (Jeff, 07-21: "if we run the sparse branch with FA2 sliding window and
    # it suddenly gets much faster, we know flex attention is a problem").
    #   flex    (default) production path: compiled flex + materialized GQA
    #   flexgqa flex with native GQA (isolates the repeat_interleave cost)
    #   fa2swa  FA2 native sliding window, NO SINK — output is knowingly
    #           WRONG for rows past the window (timing only, never for
    #           training real adapters); window via DELTA_FA2_WINDOW
    # gamma=1 identity still holds under ANY impl (every row anchored),
    # so startup_validation stays meaningful.
    impl = os.environ.get("DELTA_SPARSE_IMPL", "flex")
    if impl not in ("flex", "flexgqa", "fa2swa"):
        raise SystemExit(f"unknown DELTA_SPARSE_IMPL={impl!r} — refusing to "
                         "guess (a typo here must not silently mislabel a "
                         "diagnostic or corrupt a training run)")
    if impl != "flex":
        # diagnostic-only paths (fa2swa is KNOWINGLY WRONG math: no sink).
        # Loud on every call would spam; loud once per process:
        if not getattr(delta_forward_train, "_impl_warned", False):
            print(f"[flex_delta] WARNING: DELTA_SPARSE_IMPL={impl} — "
                  "BENCH-ONLY sparse branch, never train real adapters "
                  "with this", flush=True)
            delta_forward_train._impl_warned = True

    # GQA expansion is UNCONDITIONAL (as before this commit): the anchor
    # branch below always consumes expanded k/v via the identical sdpa
    # call, so the anchor branch is bit-identical across variants and any
    # timing difference is attributable to the sparse branch alone.
    # (07-21 review: enable_gqa TOGETHER with a materialized attn_mask
    # forces sdpa onto the math backend — slower, ~GBs of transient
    # attention weights at 32K, and only for the non-flex variants: an
    # unfair comparison and an OOM risk.)
    kr, vr = k, v
    if k.size(1) != h:
        kr = k.repeat_interleave(h // k.size(1), dim=1)
        vr = v.repeat_interleave(h // v.size(1), dim=1)

    if impl == "fa2swa":
        from flash_attn import flash_attn_func
        w = int(os.environ.get("DELTA_FA2_WINDOW", window))
        sparse = flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            causal=True, window_size=(w, 0),
            softmax_scale=scaling)  # [b, s, h, d]; GQA native
    elif impl == "flexgqa":
        bm = get_block_mask(s, int(window), int(sink), str(q.device))
        sparse = _get_flex()(q, k, v, block_mask=bm, scale=scaling,
                             enable_gqa=True)
        sparse = sparse.transpose(1, 2)
    else:
        bm = get_block_mask(s, int(window), int(sink), str(q.device))
        sparse = _get_flex()(q, kr, vr, block_mask=bm, scale=scaling)
        sparse = sparse.transpose(1, 2)  # [b, s, h, d]

    k, v = kr, vr  # anchor branch: expanded k/v, identical for ALL impls

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
    elif delta_grad_scale != 1.0:  # gradient-scale arms (1/sqrt(gamma), 1/gamma)
        delta = _ScaleGrad.apply(delta, delta_grad_scale)

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
