"""Prefill drift telemetry: delta_interanchor_cos (master plan mandatory).

During a delta-mode prefill, ``delta_forward`` materializes the anchor deltas
(``dense_sparse_diff``: dense minus sparse output at every gamma-th row). The
science signal for WP-1 (adaptive stride) and WP-3 (decode stride choice) is
how similar consecutive anchor deltas are: high cosine = the delta is smooth
and safe to reuse across the gamma block; low cosine = under-sampling.

``drift_summary`` runs on-GPU inside the forward (cheap: one cosine over an
already-materialized tensor) and returns a JSON-serializable per-layer summary.
``aggregate_drift`` is pure Python (no torch) so the runner can ingest sidecar
files and unit tests can run offline.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

HIST_BINS = 40
HIST_RANGE = (-1.0, 1.0)


def drift_summary(dense_sparse_diff, layer_idx: int, gamma: int, seq_len: int) -> Dict[str, Any]:
    """Per-layer summary of cos(anchor delta_i, anchor delta_{i+1}), per head.

    ``dense_sparse_diff``: [b, n_anchors, h, d] as materialized in
    ``LlamaAttention.delta_forward`` just before the broadcast reshape.
    """
    import torch

    a = dense_sparse_diff[:, :-1].float()
    b = dense_sparse_diff[:, 1:].float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # [b, n_anchors-1, h]
    flat = cos.flatten()
    hist = torch.histc(flat, bins=HIST_BINS, min=HIST_RANGE[0], max=HIST_RANGE[1])
    q = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9], device=flat.device))
    return {
        "layer": int(layer_idx),
        "gamma": int(gamma),
        "seq_len": int(seq_len),
        "n": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "p10": float(q[0].item()),
        "p50": float(q[1].item()),
        "p90": float(q[2].item()),
        "hist_bins": HIST_BINS,
        "hist_range": list(HIST_RANGE),
        "hist": [int(x) for x in hist.tolist()],
    }


def aggregate_drift(lines: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Aggregate per-request summaries into one entry per layer (pure Python).

    Means are n-weighted; histograms are summed (bins must match — they come
    from this module's constants, so a mismatch means mixed-version sidecars
    and is an error, not something to paper over).
    """
    out: Dict[int, Dict[str, Any]] = {}
    for s in lines:
        layer = int(s["layer"])
        if s["hist_bins"] != HIST_BINS or list(s["hist_range"]) != list(HIST_RANGE):
            raise ValueError(
                f"drift sidecar bins/range {s['hist_bins']}/{s['hist_range']} do not match "
                f"current {HIST_BINS}/{list(HIST_RANGE)} — mixed-version sidecar files")
        agg = out.setdefault(layer, {
            "layer": layer, "n": 0, "mean": 0.0, "requests": 0,
            "hist": [0] * HIST_BINS, "hist_bins": HIST_BINS,
            "hist_range": list(HIST_RANGE), "gamma": s["gamma"],
        })
        total = agg["n"] + s["n"]
        if total > 0:
            agg["mean"] = (agg["mean"] * agg["n"] + s["mean"] * s["n"]) / total
        agg["n"] = total
        agg["requests"] += 1
        agg["hist"] = [x + y for x, y in zip(agg["hist"], s["hist"])]
    return out


def hist_bin_edges() -> List[float]:
    lo, hi = HIST_RANGE
    step = (hi - lo) / HIST_BINS
    return [lo + i * step for i in range(HIST_BINS + 1)]
