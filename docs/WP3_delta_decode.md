# WP-3 — Delta Correction During Decoding

You are a coding agent in the delta-attention fork. Prereq: WP-0 merged, Gates 1–2 green. Read `00_MASTER_PLAN.md`; rules are binding. This WP adds a sparse-decode-with-delta path and its instrumentation. It is inference-only.

## Background (what exists)

Current behavior (`delta_attention/sample.py`, patched `_sample`): prefill runs under the sparse implementation with `mode∈{delta,recompute,sparse-only}`, then the code sets `self.config._attn_implementation = "sdpa_rectangle"` and ALL decode steps run dense attention over the full KV cache (`LlamaAttention.sdpa_rectangle_forward`, llama.py ~line 498). Nothing is evicted from the cache — full-attention anchor rows during decode are always possible.

## The method to implement

Add `decode_mode` ∈ {`dense` (current, default), `sparse` (floor baseline), `delta` (new)} and, for delta: `gamma_dec` (int) and `refresh_policy` ∈ {`fixed`, `drift`} with `drift_threshold` (float, cos-sim).

Per attention layer, maintain decode state: `last_delta: Tensor[b, h, d] | None`, `steps_since_anchor: int`, and (drift policy) `last_sparse_out: Tensor[b, h, d] | None`.

Each decode step (q_len == 1), per layer:
1. Compute **sparse single-row attention**: query vs {sink tokens (first 1024) + last `sliding_window` keys} of the cache. Implement as one torch SDPA call over the gathered key/value slices with correct softmax over exactly that set (sinks + window; mirror the prefill sparse pattern as closely as the cache layout allows). Keep it simple and correct; this op is tiny.
2. **Anchor decision:** anchor if `last_delta is None` or (`fixed` and `steps_since_anchor >= gamma_dec`) or (`drift` and cos_sim(sparse_out, last_sparse_out) < drift_threshold) — the drift trigger fires when the sparse output starts moving fast, a cheap proxy for delta staleness. Always also anchor at `steps_since_anchor >= gamma_dec_max` (config, default 128) so drift mode can't starve.
3. If anchor: run the existing dense rectangle for this row (`sdpa_rectangle_forward` machinery), set `last_delta = dense_out − sparse_out`, `steps_since_anchor = 0`, output `dense_out` (exact at anchors, mirroring prefill).
4. Else: output `sparse_out + last_delta`, increment counter.
5. Update `last_sparse_out`.

State lives on the attention module, reset at each new `generate` call (hook the prefill branch of `_sample`). Batch size 1 is the supported case for delta decode; assert and document.

## Instrumentation (mandatory — this is the science)

When `decode_mode=delta` and config flag `log_drift=true`, every `drift_log_every=1` steps additionally compute the TRUE per-step delta (dense row − sparse row; yes this makes instrumented runs slower — that's fine, it's a measurement mode; a `log_drift=false` path must exist for latency benchmarks) and log per layer:
- `delta_drift_cos`: cos(true_delta_t, last_anchor_delta) — logged against `steps_since_anchor`.
- `applied_vs_true_cos`: cos(applied output, true dense output).
- `anchor_rate`: anchors per 100 tokens (drift policy).
Aggregate to wandb as per-layer line series + histograms. These curves (drift vs steps-since-anchor, per layer) are the headline figure; treat their correctness as a deliverable equal to accuracy numbers.

## Tests (extend Gate 1; must pass before any eval)

- **T6 anchor-step exactness:** at an anchor step, delta-decode output == sdpa_rectangle output (cos > 0.999).
- **T7 γ_dec=1 ⇒ dense decode:** with `gamma_dec=1`, full generation token-ids on a 4K NIAH prompt must match `decode_mode=dense` token-ids exactly (greedy).
- **T8 zero-delta consistency:** force `last_delta=0` → non-anchor steps equal pure sparse decode.
- **T9 state hygiene:** two consecutive `generate` calls produce identical outputs to fresh-model calls (state reset works).

## Experiments (from experiments.yaml — do not invent configs)

Night-1 set: `t3_dense_decode` (ceiling), `t3_sparse_decode` (floor), `t3_delta_dec_g{8,16,32,64}` (fixed), `t3_delta_dec_drift_{0.90,0.95}` — all with prefill mode=delta, γ=64, window=2048, RULER lengths per YAML. Plus one latency config `t3_latency` (log_drift=false, measures decode_ms_per_token vs dense decode at 131K).

## Acceptance criteria

- T6–T9 green; Gate-2 e2e passes with `decode_mode=delta`.
- `--smoke` run of the t3 set completes; check_smoke ordering holds: dense ≥ delta_dec ≥ sparse_decode at 4K.
- Drift curves render in wandb for at least one smoke run (visual artifact linked in the PR).
- No monitoring loops. Launch → validate → report URL → done.

## Design constraints

- Do not modify prefill delta code paths.
- Do not evict KV cache entries (anchors need the full cache).
- RoPE: the query arriving at decode already has rotary applied via the normal forward path — reuse the existing flow; do not re-apply.
- GQA: cache K/V have 8 KV heads; `repeat_kv` before dense/sparse ops exactly as `sdpa_rectangle_forward` does.
