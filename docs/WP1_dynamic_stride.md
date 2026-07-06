# WP-1 — Dynamic Stride (Adaptive γ) for Prefill Delta

You are a coding agent in the delta-attention fork. Prereq: WP-0 merged, Gates 1–2 green. Read `00_MASTER_PLAN.md`; rules binding. Inference-only.

## Background

`delta_forward` (llama.py ~531) currently: anchors at `idx = arange(0, s_p, step=lambd)`; dense rows via `qsa_kernel(Q_sel, K, V, idx, scaling)`; delta broadcast via `reshape(b, s_p//lambd, lambd, h, d)` — which hard-assumes uniform stride. The Triton kernel already accepts arbitrary `idx`; only the broadcast needs generalizing.

## Task A — variable-stride correction (mechanical)

Implement `apply_delta_variable(sparse_out, dense_out_sel, idx, s_p)`:
- `deltas = dense_out_sel − sparse_out[:, idx]`
- assignment: each row i ∈ [0, s_p) receives the delta of the nearest anchor ≤ i → `owner = torch.searchsorted(idx, arange(s_p), right=True) − 1`
- `corrected = sparse_out + deltas[:, owner]` (gather; no reshape).
**T10 (Gate-1 extension):** with uniform idx (γ=64), `apply_delta_variable` output must equal the existing reshape path bit-for-bit (same dtype ops) or cos > 0.9999. This test gates everything else in this WP.

## Task B — chunked adaptive prefill

Config additions: `stride_policy` ∈ {`fixed` (default), `adaptive`}, `gamma_min` (16), `gamma_max` (256), `adapt_chunk` (4096), `adapt_threshold` (cos).
Adaptive prefill processes anchor placement in chunks of `adapt_chunk` rows:
- Within a chunk, place anchors at the current local stride γ_c.
- After computing the chunk's anchor deltas, measure mean consecutive-anchor cos sim `c`:
  - `c < adapt_threshold` → γ_next = max(γ_c/2, gamma_min)
  - `c > adapt_threshold + margin (0.02)` → γ_next = min(γ_c·2, gamma_max)
  - else keep.
- First chunk starts at γ=64.
Implementation note: the sparse pass over the whole sequence stays single-shot exactly as today; only anchor selection + the qsa_kernel call + correction become chunk-loop-driven. Keep the existing `cut_n` dense tail logic untouched (it sits outside the adaptive region).
Log per chunk: chosen γ, mean/min inter-anchor cos, cumulative anchor count. Report **effective sparsity** = 1 − (window·s + Σ anchor_row_costs)/(s²/2) in results.csv (add column `effective_sparsity`; also add it for fixed-γ configs so frontiers are comparable).

## Task C — (ablation, lower priority) global difficulty predictor

Only after A+B are merged and evaluated: a heuristic global-γ picker from prompt statistics (length, task marker) — implement as a config `stride_policy=heuristic` with a simple lookup. Do not build a learned model in this WP.

## Tests

- T10 (above).
- **T11 adaptive-degenerate:** with `gamma_min=gamma_max=64`, adaptive path output == fixed-γ64 path output exactly.
- **T12 anchor exactness holds under variable idx** (reuse T2 logic with a random non-uniform idx).

## Experiments (experiments.yaml only)

`t1_fixed_g{32,64,128,256}` (frontier baseline; g64 shared with base) and `t1_adaptive_thr{0.90,0.95,0.98}`. Deliverable figure: accuracy (RULER avg per length) vs effective_sparsity, adaptive points overlaid on the fixed-γ frontier. Success = adaptive point above the fixed frontier at ≥1 operating point at 65K or 131K.

## Acceptance criteria

- T10–T12 green; smoke run of t1 set completes with ordering sanity.
- effective_sparsity present for every t1 row.
- No monitoring; no invented configs; prefill numerics for fixed-γ configs unchanged vs base (guarded by T11 + Gate-3 re-run).
