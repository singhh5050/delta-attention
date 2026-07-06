# WP-0 — Fork, Eval Harness, Validation Gates, Smoke Tests

You are a coding agent working in a fork of https://github.com/jeffwillette/delta-attention. Read `00_MASTER_PLAN.md` first; its rules of engagement are binding. Your job in WP-0: make the repo runnable end-to-end, build the experiment runner, and implement the three smoke gates. You do NOT run any full-scale experiment in this WP.

## Deliverables

1. `env/setup.sh` — reproducible environment install (CUDA 12.x, torch 2.8.0, triton 3.4.0, `pip install -r requirements.txt`, hip-attn 1.2.9). Must end by running Gate 1 and printing PASS/FAIL per test.
2. `eval/run_matrix.py` — the experiment runner. Reads `experiments.yaml`, executes the configs whose names are passed via `--configs` (comma-list or `all`), supports `--smoke`. For each config: starts `server_hf.py` with the config's model settings, runs the RULER harness against it for the config's tasks/lengths/samples, tears down, appends one row to `results/results.csv`, logs to wandb. Sample-level parallelism across visible GPUs (one server per GPU, shard samples).
3. `eval/ruler_client.py` — thin client around NVIDIA RULER (vendored as a git submodule or pinned clone) pointing at the local server. Use RULER's own data generation; pin its version in a lockfile.
4. `delta_attention/validation.py` — startup validation gate (§4) importable by every entrypoint.
5. `tests/test_math_identities.py` — Gate 1 (§3).
6. `eval/smoke_e2e.py` — Gate 2 (§5).
7. `results/` — results.csv schema per master plan; one row per config run.
8. wandb: project name and entity read from env vars `WANDB_PROJECT` / `WANDB_ENTITY`. Every run logs the mandatory metric set from the master plan.

## §1 Environment notes

- `hip-attn==1.2.9` builds Triton kernels; verify import with a minimal `block_sparse_attention` call on random tensors before declaring the env good.
- HF auth: expect `HF_TOKEN` env var with gated access to `meta-llama/Llama-3.1-8B-Instruct`. Fail fast with a clear message if the model card is inaccessible.
- Pin everything; do not upgrade `transformers` (llama.py copies its internals against 4.51.3).

## §2 Runner contract

- A config row fully specifies: `name, mode, attn_implementation, delta_lambda, sliding_window, decode_mode, gamma_dec, refresh_policy, drift_threshold, context_lengths, tasks, n_samples, seed, max_new_tokens`.
- `--smoke` applies `smoke_overrides` from the YAML (context_lengths→[4096], tasks→smoke task list, n_samples→50). Nothing else about the code path changes.
- Runner never retries a failed config silently. On failure: record `status=failed` + error string in results.csv, continue to next config.
- Runner prints, at start, the resolved config table and estimated sample count; at end, a summary table. No interactive prompts.

## §3 Gate 1 — math identity tests (tests/test_math_identities.py)

All tests use `meta-llama/Llama-3.1-8B-Instruct` attention modules with random inputs where possible, bf16, on 1 GPU, seq lengths small (2048–8192) so dense reference is cheap. Tolerances: report max abs diff and cos sim; pass thresholds stated per test. Structure as pytest.

- **T1 qsa_kernel correctness.** Random Q,K,V (b=1, h=32, s=4096, d=128), arbitrary sorted idx (e.g., 37 rows incl. 0 and s-1). Compare `qsa_kernel` output rows against a causal torch-SDPA reference restricted to those query rows. PASS: cos sim > 0.999 per row, max abs diff consistent with bf16 (<2e-2).
- **T2 anchor exactness.** Full model single layer path: run `delta_forward` (mode=delta, window=2048, γ=64) and `sdpa` dense on identical inputs (s=8192). At anchor rows (idx multiples of γ within the non-cut region) corrected output must match dense output. PASS: cos sim > 0.999 at anchors. (Rows inside the `cut_n` tail block are dense by construction — also check them.)
- **T3 γ=1 ⇒ dense everywhere.** Same setup with `delta_lambda=1`. Corrected output must match dense at ALL rows. PASS: mean cos sim > 0.999, p1 cos sim > 0.995. This is the strongest end-to-end indexing test; if it fails, nothing downstream is trustworthy.
- **T4 window ≥ context ⇒ delta ≈ 0.** `sliding_window=s` (window covers everything, plus 1024 sinks). The computed delta tensor must be ~0 (‖Δ‖/‖output‖ < 1e-2) and corrected output ≈ dense.
- **T5 full-model logit sanity.** Whole 32-layer model, one 4K prompt: logits under (mode=delta, γ=1, window=2048) vs dense FA2 — top-1 next-token prediction must agree on ≥ 95% of positions. (Catches layer-wiring bugs T2–T4 can miss.)

Any Gate-1 failure: stop, write a failure report (test, tensors' summary stats, suspected file/line), do not proceed to Gate 2.

## §4 Startup validation gate (delta_attention/validation.py)

Called by every entrypoint (server, runner, training) within its first 60s:
1. Assert config row resolves with no missing/extra keys.
2. Run T2 (anchor exactness) as a fast self-check at s=2048 with the *live* config's window/γ (skip when γ makes it meaningless, e.g., recompute mode — document which modes skip which checks).
3. Generate on one built-in known-answer NIAH prompt (needle = fixed UUID at depth 50% of a 2K filler); assert the UUID appears in the generation for modes {delta, dense}; for sparse-only just assert non-empty output.
4. Assert every mandatory wandb key has been logged at least once (log zeros/nulls at init).
Failure at any step: print reason, `sys.exit(1)`.

## §5 Gate 2 — single-sample E2E (eval/smoke_e2e.py)

Through the REAL server + REAL RULER client (no shortcuts): one `niah_single_1` sample at 4096 and one at 32768, for configs `base_delta_g64`, `base_sparse_only`, `base_dense_fa2` (from experiments.yaml). Asserts: server healthy; generation non-empty; needle retrieved for delta and dense at both lengths; results.csv rows appended; wandb runs created with all mandatory keys. Total budget: ~10 min on 1 GPU.

## §6 Gate 3 — mini-matrix

Not a separate script: `python eval/run_matrix.py --configs night1_all --smoke`. Smoke settings: 4K only, tasks = {niah_single_1, niah_multikey_2, qa_1}, 50 samples, seed 0. Post-run assertion script `eval/check_smoke.py` verifies:
- Every night-1 config has a results row with status=ok.
- Ordering sanity at 4K: acc(dense) ≥ acc(delta) ≥ acc(sparse_only) − 5pt.
- `base_delta_g64` and `base_sparse_only` within ±10pt of paper 4K anchors (96.5 / 90.5) on the overlapping tasks (tolerance is wide: 50 samples is noisy).
- Per-config wall time recorded → print extrapolated full-run node-hours for the night-1 matrix.

## Acceptance criteria for WP-0

- Fresh instance → `bash env/setup.sh` → Gate 1 all PASS, no manual steps.
- Gate 2 passes on 1 GPU.
- Gate 3 passes on ≤ 8 GPUs in ≤ 90 minutes and prints the extrapolation table.
- README section documenting: how to add a config, how to run smoke vs full, where results land.

## Forbidden

- Monitoring running jobs; sleeping/polling loops longer than server-startup healthchecks.
- Editing `experiments.yaml` values (adding the file from the provided template is fine).
- Changing kernels, model math, or the `assert h == 32`.
- Upgrading pinned deps.
