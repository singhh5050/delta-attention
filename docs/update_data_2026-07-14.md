# Data record: 2026-07-14 runs (post "32K reversal" update to Jeff)

Everything run since `docs/update_data_2026-07-12.md`. Four boxes, all
self-terminated after wandb archive: "harsh experiment 5" (train32k, ~$30),
"6" (distill, ~$10), "7" (enmc, ~$6), "8" (distill2, ~$12). Code:
`master@3dcd6a1` (boxes 5–7) and `master@4c25f0b` (box 8). All perplexity
evals use the SAME deterministic 32 held-out PG19-test chunks @32,768 tokens
as the 07-12/07-13 runs, so paired comparisons hold across runs.

Review note: commit 70d35db (the new modes) went through a high-effort
adversarial code review before launch; it caught a launch-blocking bug
(training loss at >8K routed through hip's memory_efficient_llm_ce, which is
FORWARD-ONLY — backward() would have raised at step 1) plus a dead ~2.1GB
causal-mask allocation per 32K forward. Fixes in 3dcd6a1; the chunked-CE/KL
loss functions are validated against their unchunked forms (and backprop)
in startup_validation on every run.

---

## 1. Training at 32K (train32k, box 5)

**Design:** retrain the three pilot arms (delta / dense / detach) at
seq_len 32,768 — 500 steps, so the TOKEN budget equals the 8K pilot
(2000 × 8192 = 500 × 32768); sequence length is the only lever moved.
Same LoRA (r16 α32 q/k/v/o), lr 1e-4→1e-5 cosine, shuffled PG19, same seed.
CE via chunked_ce (per-chunk checkpointed fp32; validated vs full CE at
startup). Artifacts: `wp2_adapter_{delta,dense,detach}_32k:latest`.
Wall-clock per arm: delta 54 min, dense 68 min, detach ~53 min.
Training runs: delta `etoxgd7p`; final delta loss 2.2574.

### 1a. Pipeline eval @32K (run `1e777qkb`, 32 chunks)

| arm | ppl (32K-trained) | ppl (8K-trained, 07-12) |
|---|---|---|
| base | 12.2219 | 12.2219 |
| delta-trained | 10.4067 | 10.3732 |
| detach-trained | 10.5474 | 10.4691 |
| dense-trained | 10.6680 | 10.5851 |

Paired per-chunk loss diffs (32K-trained arms):

- delta_32k − dense_32k = **−0.0248 ± 0.0014 (sem), better on 32/32 chunks**
  (8K-trained gap was −0.0202 ± 0.0011 → the delta-specific gap GREW ~23%
  when training at the length where the mechanism is active)
- delta_32k − detach_32k = −0.0134 ± 0.0006, 32/32
- detach_32k − dense_32k = −0.0114 ± 0.0010, 32/32
- Monotone gradient-flow dose-response (delta < detach < dense) replicates,
  every pairwise gap significant, every chunk in the predicted direction.

Absolute ppl is slightly WORSE than the 8K-trained arms across the board
(+0.03–0.08): equal tokens but 4× fewer optimizer steps (500 vs 2000).
The delta-specific contrast is what training length improved.

### 1b. Dense eval @32K — crossover replicates (run `s9rxrd1m`)

| arm | dense-eval ppl |
|---|---|
| base | 11.2652 |
| dense-trained | 9.7907 |
| detach-trained | 9.9365 |
| delta-trained | 9.9616 |

delta_32k − dense_32k = **+0.0173 ± 0.0010, delta better on 0/32 chunks** —
the sign flips on every single chunk. Stronger crossover than the 8K arms
(+0.0063). Each model is best under the attention path it trained with.

### 1c. Pipeline tax (loss penalty of pipeline vs dense eval, same model)

| arm | tax | vs base |
|---|---|---|
| base | 0.0815 | — |
| dense_32k-trained | 0.0858 | +5% |
| detach_32k-trained | 0.0597 | −27% |
| delta_32k-trained | **0.0437** | **−46%** |

(8K delta-trained was 0.0574 / −30%.) Training through the pipeline at 32K
is the strongest tax treatment measured, while keeping full PG19 adaptation.

### 1d. Anchor-gradient decomposition — Jeff's 1/γ question (run `etoxgd7p`)

Probe: layer-0 query gradients through delta_forward_train (γ=64) on a
held-out 32K batch, every 50 steps (10 probes). Arm-faithful probe (post
07-13 fix), plus the detached-channel decomposition:

- anchor_grad_ratio (full graph): **mean 3.405, range 3.373–3.427** — flat
  across training. (At 8K this quantity was 1.46; concentration grows with
  sequence length.)
- anchor_grad_ratio_detached (correction term detached): **0.955** —
  i.e. ~1.0, NO anchor concentration at all.

Reading: essentially ALL anchor-row gradient concentration flows through the
γ-times-summed correction term (Jeff's mechanism confirmed), but its
magnitude is ~3.4× at 32K, nowhere near γ=64. And the dose-response above
says more of this signal → better model, so the concentration appears
beneficial; no evidence a 1/γ backward correction is needed.

---

## 2. Distillation pilot (distill, box 6)

**Design:** 4th arm, same dials as the CE pilot (2000 steps @8K, same
data/seed): student = base + LoRA under the differentiable delta pipeline;
teacher = the SAME weights with adapters disabled under dense (sdpa)
attention; loss = per-token full-vocab KL(teacher ‖ student), fp32
per-chunk-checkpointed. Pure KL (distill_alpha=0). Training: 57 min.
Artifact `wp2_adapter_distill:latest`.

### 2×2 @32K, 5 arms (pipeline `nm8z7hub`, dense `dcatrv6o`, same 32 chunks)

| arm | pipeline ppl | dense ppl | tax |
|---|---|---|---|
| base | 12.2219 | 11.2652 | 0.0815 |
| dense-trained (CE) | 10.5851 | 9.7333 | 0.0838 |
| detach-trained (CE) | 10.4691 | 9.7738 | 0.0687 |
| delta-trained (CE) | 10.3732 | 9.7946 | 0.0574 |
| **distill-trained** | **11.7148** | **11.1852** | **0.0463** |

Paired: distill − base = −0.0424 ± 0.0020 under pipeline (32/32 chunks),
−0.0071 ± 0.0006 under dense (32/32).

Reading: pure KL does exactly and only its designed job. Largest tax cut of
the 8K-trained arms (0.0815 → 0.0463, −43%), but absolute ppl barely better
than base — structurally: the teacher IS the base model, so KL anchors the
student to base-dense behavior and forfeits PG19 adaptation (dense-eval ppl
11.19 ≈ base 11.27 confirms it learned "close the pipeline gap" and nothing
else). CE and KL capture complementary effects → distill2 follow-up (§4).

---

## 3. InfiniteBench En.MC (enmc, box 7)

**Design:** longbook_choice_eng (229 book-comprehension MCQs, 4 options),
official template, scored by argmax over the A/B/C/D letter logprobs at the
last prompt position — the MMLU-style low-noise scoring, at a context length
where delta ≠ dense. Book contexts are ~100K+ tokens: middle-truncated
(InfiniteBench's own head+tail scheme) to 31,500 tokens to match the 32K
regime of every other measurement. NOT comparable to published 128K En.MC
numbers; internally valid across arms (identical truncated inputs).
Whole eval (229 × 4 arms, prefill-only): ~25 min. Run `44cqxfpv`
(smoke `3q0bgb1e`).

| arm | accuracy (n=229, chance 25%) |
|---|---|
| base + dense | 0.6114 |
| ce_delta (8K delta-trained) | 0.5677 |
| dense_delta (8K dense control) | 0.5590 |
| base + delta | 0.5546 |

Readings:
1. Instrument works: 55–61% vs the 25% floor (LongBench-v2 MCQ sat at
   chance and could distinguish nothing).
2. Pipeline cost ≈ 5.6 pts (61.1 → 55.5) — real, modest, no RULER-style
   collapse (consistent with the LongBench QA finding).
3. Training effect directionally consistent with the ppl result
   (ce_delta > dense_delta > base_delta under the pipeline: +1.3 / +0.9 pts)
   but WITHIN NOISE — binomial sem ≈ 3.3 pts at n=229. A null with the right
   sign, not evidence.
4. Instrumentation gap: only aggregate accuracy logged, no per-sample
   correctness → no paired (McNemar) test possible post-hoc. Add per-sample
   dump before rerunning En.MC on the 32K adapters.

---

## 4. Distill follow-up (distill2, box 8) — IN FLIGHT at time of writing

Two arms, same 8K pilot dials, code `4c25f0b`:

- **mix** (`--distill-alpha 1.0`, tag `_mix`): KL-to-base-teacher + CE.
  Does one objective capture adaptation AND the tax cut? (Tension: the KL
  anchor is the UNadapted base; CE pulls away from it.)
- **dft** (`--teacher-checkpoint checkpoints/pilot_dense`, tag `_dft`):
  pure KL to the DENSE-FINETUNED teacher — a separate frozen 16GB model with
  the pilot_dense LoRA merged in (init_model's merge path; deliberately NOT
  PEFT set_adapter switching, which toggles requires_grad on the student).
  Student still starts from plain base → adapter evals like any arm.
  Prediction: if distill's tax (~0.046) transfers onto dense-ft (dense-eval
  9.73), pipeline ppl ≈ 10.19 < delta-CE's 10.37 → would be the best
  pipeline number in the project.

Chain ends with the 7-arm 2×2 @32K (arms base, delta, dense, detach,
distill, distill_mix, distill_dft; both forwards; same 32 chunks).
Results to be appended.

---

## Open items

- Rotate Lambda/HF/wandb keys (still pending; current keys traveled to
  boxes 5–8, all now terminated).
- Per-sample logging for MCQ suites, then optionally En.MC on the 32K
  adapters (`ce32k_delta` etc. already wired in longbench_eval.py).
- LongBench QA on the 32K adapters (cheap, same wiring).
- InfiniteBench at full 128K budget (different experiment; decouples from
  the 32K grid).
