# Delta Attention Extensions — Master Plan

Extension of Willette et al., "Delta Attention: Fast and Accurate Sparse Attention Inference by Delta Correction" (NeurIPS 2025, arXiv:2505.11254). Base repo: https://github.com/jeffwillette/delta-attention (fork it; all work happens in the fork).

## The three research tracks

1. **WP-1 Dynamic stride**: replace fixed γ=64 with adaptive anchor placement driven by measured delta drift. Claim: better accuracy-vs-sparsity frontier than any fixed γ.
2. **WP-2 Trainable delta**: fine-tune the model with the delta pipeline active so gradients flow through the reused correction, pressuring anchor tokens to become summary representations. Claim: delta-smoothness becomes a trained property; corrected outputs move closer to full-attention outputs.
3. **WP-3 Delta decode**: extend delta correction from prefill into token-by-token generation (sparse decode + periodically refreshed cached delta). Claim: near-dense decode quality at near-sparse decode cost.

WP-0 is shared infrastructure and gates everything else.

## Non-negotiable rules of engagement (apply to every WP)

1. **Agents never monitor runs.** A launch task ends at: process started + startup validation passed + wandb URL printed + config name recorded. Watching a run is not a task. Never write, plan, or claim to "monitor" anything. Alerting is wandb's job (human sets alerts).
2. **Startup validation is code, not judgment.** Every experiment entrypoint runs a validation block in its first ~60s (see WP-0 §4). Any failure → print reason, `sys.exit(1)`. No experiment may skip the gate. A run that dies at the gate is a success of the system, not a failure to hide.
3. **The experiment matrix is data, not discretion.** All runs come from `experiments.yaml` rows. Agents implement the runner that executes rows; agents never invent configs, change sample counts, or "try a quick variant." If a config seems wrong, stop and report — do not improvise.
4. **Smoke = same pipeline, smaller numbers.** The smoke test is the real runner with `--smoke`, which shrinks context/tasks/samples per the YAML `smoke_overrides`. There are no separate smoke scripts. Gate order: Gate 1 (math identities) → Gate 2 (single-sample E2E) → Gate 3 (mini-matrix). All green before any full-scale run.
5. **One results sink.** Every completed run appends exactly one row to `results/results.csv` (and mirrors to a wandb table), keyed by config name. No row = it didn't happen.
6. **No silent fallbacks.** If a kernel, import, or shape assumption fails, crash loudly. Never fall back to a different attention implementation without an explicit config flag.
7. **Determinism where possible.** Fixed seeds in configs; greedy decoding (temperature=0) for all evals, matching the repo default.

## Repo facts every agent must know (verified against the actual code)

- `delta_attention/config.py`: `Config` dataclass. Key fields: `attn_implementation` ∈ {window, hip_attention, flash_attention_2}; `mode` ∈ {delta, recompute, sparse-only}; `delta_lambda` (γ, default 64); `sliding_window` (default 2048). "window" = Streaming-LLM-style (sink 1024 + sliding window) implemented via hip-attn's `block_sparse_attention`.
- `delta_attention/llama.py`: modified HF Llama. `LlamaAttention.delta_forward` (~line 531) is the core: runs sparse attention for all rows, selects anchor indices `idx = arange(0, s_p, step=lambd)`, computes dense rows via `qsa_kernel(recomp_query_states, K, V, idx, scaling)` from `delta_kernel.py`, takes the difference at anchors, reshapes `(b, s_p//lambd, lambd, h, d)` and broadcasts the delta. Note `cut_n = s % lambd + max(128, lambd)`: a dense block at the sequence end is always recomputed (paper Appendix C). **`assert h == 32`** — hardcoded to Llama-3.1-8B. Do not generalize; do not remove the assert.
- `delta_attention/delta_kernel.py`: Triton query-sparse FA2 kernel, **forward only**. Accepts arbitrary query index tensor `idx` — variable strides need NO kernel changes.
- `delta_attention/sample.py`: monkey-patched HF `_sample`. Prefill runs under the configured sparse implementation with `mode`; then it flips `self.config._attn_implementation = "sdpa_rectangle"` so **all decoding is currently fully dense** (new query vs entire KV cache via `sdpa_rectangle_forward` in llama.py, ~line 498). This flip is WP-3's insertion point.
- `server_hf.py` + `model_wrapper.py`: OpenAI-style HTTP server intended to be hit by NVIDIA's RULER harness (https://github.com/NVIDIA/RULER).
- Pins: `transformers==4.51.3`, `hip-attn==1.2.9`, `torch==2.8.0`, `triton==3.4.0`. Do not upgrade transformers (llama.py copies its internals).
- Model: `meta-llama/Llama-3.1-8B-Instruct` only. All work assumes it.

## Paper reference numbers (for validation tolerances)

RULER avg, Llama-3.1-8B-Instruct, window=2048, γ=64 — from Table 1:
- 4K: FA2 96.74 | Str.LLM 90.52 | Str.LLM+Δ 96.54
- 32K: FA2 85.84 | Str.LLM 30.25 | Str.LLM+Δ 81.27
- 131K: FA2 73.16 | Str.LLM 27.45 | Str.LLM+Δ 64.40

## Sequencing

1. WP-0 (infra + Gates 1–3). Blocks everything.
2. WP-3 and WP-1 in parallel (inference-only; share drift instrumentation from WP-0).
3. WP-2 (heaviest; needs the differentiable path + equivalence gate).
4. Night-1 full matrix per `experiments.yaml` once Gates are green.

## Hardware assumptions

Single H100 80GB fits 8B + 131K KV cache (~35GB total). Eval sweeps shard samples across one 8×H100 node. Training (WP-2) uses its own separate 8×H100 instance. Never co-tenant training and evals.

## Mandatory wandb metrics (all experiments)

`config_name, mode, gamma, gamma_dec, refresh_policy, context_len, task, n_samples, accuracy, samples_per_sec, prefill_ms_p50, decode_ms_per_token_p50, oom_count`
Plus instrumentation (logged during runs, not post-hoc):
- prefill: `delta_interanchor_cos` (cos sim between consecutive anchor deltas, per layer, logged as histogram) — feeds WP-1.
- decode (WP-3 modes): `delta_drift_cos` vs `steps_since_anchor` (per layer) — the key science metric.
