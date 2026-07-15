# Data record: 2026-07-15 specdec run (Jeff's γ-sweep + GovReport asks)

Box "harsh experiment 9" (1×H100, us-south-2), chain mode `specdec`,
code `master@a5151d9`, self-terminated after archive. Answers Jeff's
8:53 PM message: γ_dec between 2 and 16 on the LongBench numbers, and
GovReport as the speculative-decoding probe. Wandb runs: QA sweep
`rwiomfcx`, GovReport `23o7tkqf`; per-sample scores in
`box_final_state` → `longbench_samples.csv` (new instrumentation —
paired stats below computed from it).

Fidelity note: a pre-launch review verified the sweep is comparable to the
07-13 decode numbers (same 4 QA tasks, templates, truncation, n=50, delta
γ=64 prefill; ONLY gamma_dec varies), GovReport uses LongBench's official
template/512-token budget/ROUGE-L-f, and our sparse decode is sink+window
only (no other retrieval path), so Jeff's "RULER is impossible for the
draft" argument is valid against our implementation.

## 1. QA γ_dec sweep (LongBench v1, mean F1 over hotpotqa/2wikimqa/
   musique/multifieldqa_en, n=50/task)

| decode | mean F1 |
|---|---|
| dense (07-13) | 0.454 |
| delta γ=2 (07-13) | 0.435 |
| **delta γ=4 (new)** | **0.4329** |
| **delta γ=8 (new)** | **0.4198** |
| delta γ=16 (07-13) | 0.415 |
| pure sparse (07-13) | 0.417 |

Per-task (new arms): dec4 hotpotqa .446 / 2wikimqa .410 / musique .301 /
multifieldqa .574; dec8 .420 / .410 / .292 / .556.

- γ=4 ≈ γ=2 (.4329 vs .435): doubling correction reuse is ~free on QA.
- Smooth monotone decline 2→16, total ~2 F1; γ≥8 blends into the sparse
  floor. NO cliff — RULER's γ=2 collapse is retrieval-specific, confirmed
  with the full curve.

## 2. GovReport (LongBench gov_report, ROUGE-L f, n=50, 512 new tokens,
   delta γ=64 prefill for all non-dense arms)

| arm | ROUGE-L | paired vs sparse_dec |
|---|---|---|
| base_dense (all dense) | 0.3470 | — |
| base_delta (dense decode) | 0.3476 | — |
| delta γ_dec=2 | 0.3296 | +0.0229 ± 0.0052, 36/50 |
| delta γ_dec=4 | 0.3218 | +0.0151 ± 0.0055, 33/50 |
| delta γ_dec=8 | 0.3164 | +0.0096 ± 0.0054, 29/50 |
| delta γ_dec=16 | 0.3128 | +0.0061 ± 0.0046, 28/50 |
| sparse decode | 0.3068 | — |

Other paired contrasts: base_delta − base_dense = +0.0006 ± 0.0044 (26/50)
— delta PREFILL is exactly free on summarization, now as a paired claim.
base_delta − delta_dec4 = +0.0258 ± 0.0056 — the residual decode gap is
real.

Readings:
1. All cost is in decode; prefill sparsity costs nothing here.
2. Unlike QA, delta decode beats pure sparse at EVERY γ — significant at
   γ=2 (4.4 sem) and γ=4 (2.7 sem), fading to noise by γ=16. Recovery of
   the sparse→dense gap: γ2 ~56%, γ4 ~37%, γ8 ~24%, γ16 ~15%.
3. For the speculative frame: the delta draft is strictly better than a
   sliding-window draft at useful reuse lengths (γ=4–8), degrading smoothly
   — consistent with "easy tokens are locally predictable; hard tokens need
   the dense target."

## Review-driven instrumentation added this run

Per-sample score CSV for v1/govreport/MCQ suites (`*_samples.csv`,
archived), bare-except around the rouge scorer (RecursionError on
degenerate generations), rouge==1.0.1 pinned in requirements.txt.
Deferred from the fidelity review (deliberately): adaptive drift-refresh
arms and per-token draft-vs-dense agreement telemetry — superseded by the
true speculative decoding implementation (draft=delta, verify=dense) in
flight on the parallel workstream (specdec2 mode, e1fd1ed).

## Not run / open

- Acceptance-rate (per-step draft-argmax == dense-argmax) measurement:
  covered by the specdec2 implementation rather than telemetry.
- Key rotation still pending.
