# Complete data record since the 2026-07-09 update to Jeff

Everything measured between the Slack update (Harsh → Jeff, 2026-07-09 12:52 AM)
and 2026-07-12. Raw numbers, methods, and observations; no selection. All wandb
run IDs are in project `singhh5050-stanford-university/delta-attention`.

Model everywhere: `meta-llama/Llama-3.1-8B-Instruct`, bf16, greedy decoding
(temperature 0). Pipeline defaults unless stated: StreamingLLM-style sparse mask
(sink 1024 + sliding window 2048), delta prefill γ=64, dense decode
(`sdpa_rectangle`). Pins: transformers 4.51.3, hip-attn 1.2.9, torch 2.8.0,
flash-attn 2.8.3. All GPU runs on single Lambda H100s (one early run on the
prior 40GB A100 box, noted).

Adapters referenced throughout come from the WP-2 CE pilot (2026-07-08): LoRA
r=16, α=32 on q/k/v/o projections, lr 1e-4 cosine-decayed to 1e-5, 2,000 steps,
seq len 8192, shuffled PG19 train split (buffer 64, ≤4 chunks/doc), identical
batches and seed across three arms differing only in the training-time
attention path:

- **delta** — differentiable delta pipeline (FlexAttention `delta_forward_train`,
  T13-verified >0.999 cosine match to the inference kernel); gradients flow
  through sparse attention AND the broadcast delta correction.
- **dense** — plain dense attention (finetuned control, NOT the base model).
- **detach** — identical forward to delta, but the delta correction is
  `.detach()`ed, so no gradients flow through the correction term.

Adapters stored as wandb artifacts `wp2_adapter_{delta,dense,detach}:latest`.

---

## 1. Verification of the delta-decode mechanism (Jeff's specific asks, 07-09/07-10)

Jeff: "make sure the delta is being saved properly for every layer and reset on
the next γ-th token."

**Per-layer state.** Each of the 32 `LlamaAttention` modules owns an
independent `_dec_state` dict (cached delta tensor + step counter) and
`_dec_drift_points` telemetry list. GPU test
`tests/test_delta_decode.py::test_decode_state_is_per_layer_and_refreshes_on_gamma`:
with γ_dec=2, steps layer 0 three times → anchor flags [True, False, True],
step counters [0, 1, 0]; layer 1's state is None until layer 1 itself steps;
after layer 1 steps, its state is a distinct object with its own delta and
layer 0's state is untouched. Decode telemetry from real runs logs 32 distinct
per-layer drift series.

**Reset on schedule.** Across all 753 logged anchor events in the 32K decode
runs, anchors fired at exactly steps-since-anchor ∈ {0, γ_dec} (never off by
one). At every anchor step, cos(applied output, fresh dense output) =
1.000000 — only possible if the cached delta was replaced at that step.
Additional checks previously reported: γ_dec=1 reproduces dense decode
(T7 token-identical generations; RULER 78.0 vs dense 76.7); T6 anchor-step
output equals dense rectangle output (cos > 0.999); T8 zero-forced delta
equals pure sparse decode (exact tensor equality); T9 no state leakage across
`generate` calls (three consecutive generations identical).

Prior falsification context (pre-07-09, for completeness): γ_dec sweep at 32K
was γ=1→78.0, 2→28.7, 4→26.7, 8→26.7, 16→24.7, 64→25.3; dense 76.7; pure
sparse decode floor 20.0.

---

## 2. Post-training evaluation (t2eval chain) — run twice, replicated exactly

Method: adapters fetched from wandb, merged into the served model via
`PeftModel.from_pretrained(...).merge_and_unload()` inside `init_model`
(eval/ppl_eval.py + eval/run_matrix.py). RULER at 32,768 context, fixed 3-task
subset (niah_single_1, niah_multikey_2, qa_1), 50 samples/task = 150 prompts
per number, string_match scoring, meta-llama3 template, RULER pinned at commit
38da79d. Seed 0, greedy, max_new_tokens 128.

The full chain ran twice end-to-end on two independent boxes (first: 40GB A100
box 25dad994, 2026-07-10 morning; second: 80GB H100 box cb303dba, 2026-07-10
evening — duplicate caused by a monitoring misread, kept as a free
replication). Run IDs first/second listed as pairs.

### 2a. Held-out perplexity, 8K chunks (Jeff's #2)

16 PG19 **test-split** chunks @ 8192 tokens (≤2 chunks/doc), identical chunks
across arms, computed under the differentiable pipeline forward
(`flex_delta_train`, no grad), γ=64, window 2048.
Runs: sw6nj5t5 / w07g7654.

| arm | loss (run1/run2) | ppl (run1/run2) |
|---|---|---|
| base (no adapter) | 2.5431 / 2.5429 | 12.718 / 12.716 |
| delta-trained | 2.4083 / 2.4085 | 11.115 / 11.117 |
| dense-trained | 2.4105 / 2.4105 | 11.140 / 11.139 |
| detach-trained | 2.4091 / 2.4092 | 11.124 / 11.125 |

Observation at the time: all three trained arms improve ~12.6% and are
statistically indistinguishable from each other → read as generic PG19
adaptation, nothing delta-specific ("triple-null" at 8K). Replicate-to-
replicate noise ≈ 0.002 ppl.

### 2b. RULER 32K, trained vs base (Jeff's #2)

Runs (first/second): p32_base_g128 38t7alwi/lrsjeuyv; p32_t2ce_g64
bpaa1k9e/h6jeklsy; p32_t2ce_g128 zayce053/x1s7jg37.

| config | accuracy run1 | accuracy run2 | effective sparsity |
|---|---|---|---|
| base + delta γ64 (earlier run, same subset) | 73.3 | — | 0.798 |
| base + delta γ128 | 67.3 | 68.7 | 0.806 |
| CE-finetuned + delta γ64 | 68.0 | 68.7 | 0.798 |
| CE-finetuned + delta γ128 | 61.3 | 62.0 | 0.806 |

(Effective sparsity = 1 − computed attention pairs / full causal pairs,
causally clipped; γ64→0.798, γ128→0.806 at 32K.)

Observation: CE finetuning **costs** ~5–7 points of RULER retrieval at both
strides. Jeff's response: expected — finetuning damages the post-training/RL.
Replication delta ≤ 1.4 points on 150 prompts.

### 2c. Delta decode with the finetuned model (Jeff's suggestion, 07-10)

"It is also possible that in order for #3 to work, it has to use the #2
finetuned model." Configs p32_t2ce_dec_g2 / p32_t2ce_dec_g16: CE-delta adapter
merged, delta prefill γ=64, delta decode with fixed refresh. Same RULER subset.
Runs: hee7dlz9/yrsio4jr (γ_dec=2), e8vl2rvn/23ko7fw5 (γ_dec=16).

| config | run1 | run2 | base-model comparator |
|---|---|---|---|
| CE-ft, γ_dec=2 | 22.0 | 22.0 | 28.7 |
| CE-ft, γ_dec=16 | 20.0 | 20.0 | 24.7 |

Reference points: dense decode 76.7, pure sparse decode floor 20.0.

Observation: the CE-finetuned model does NOT rescue delta decode; it is
marginally worse than base and sits at the sparse floor at γ_dec=16.
Hypothesis refuted **for the CE-trained adapter**; a distillation-trained
model remains untested. Both numbers replicated exactly across boxes.

---

## 3. Anchor-gradient measurement (Jeff's 1/γ backward concern, 07-10)

Jeff: "the gradient to each γ-th row might be too strong because it will be
summed over γ times... we might need to inject 1/γ into the backward."

**Method.** `probe_anchor_grad_ratio` (delta_attention/train/train_delta.py):
every 100 training steps, take a held-out 8192-token batch, compute layer-0
q/k/v from the current model weights (post-layernorm, post-RoPE), run
`delta_forward_train(q, k.detach(), v.detach(), γ=64, window=2048)` with
`q.requires_grad`, backprop `out.float().pow(2).mean()`, and report
‖grad at anchor rows‖ / ‖grad at non-anchor rows‖ (row-count normalized).
Note the scope: **query gradients, layer 0, pre-projection** — not k/v, not
all layers.

**Result.** Ratio ≈ 1.46× and flat across all 2,000 steps:
delta arm (run atyhqiir): 20 probe points, min 1.457, max 1.476, mean 1.466.
detach arm (run 2hepajmg): min 1.450, max 1.465, mean 1.458.
Nowhere near γ=64.

**Instrumentation bug found and fixed (07-12).** The probe as run during the
pilot did NOT pass the arm's `detach_delta` flag — it always measured the
full-graph quantity. Therefore the detach-arm curve matching the delta-arm
curve is an artifact and says nothing about arm differences (do not quote
delta-vs-detach agreement from the pilot). The training forwards themselves
were arm-correct (verified: `flex_delta_train_forward` honors `detach_delta`);
only the diagnostic was wrong. Fixed on master: the probe now mirrors the
arm's detach setting, and delta-arm runs additionally log
`anchor_grad_ratio_detached` (same probe with the correction detached) so the
next training run decomposes the 1.46 into (γ-fold summed correction) vs
(anchor rows being the only key-dense rows). This decomposition is the direct
test of whether a 1/γ backward scaling targets anything significant.
Figure: figures/jeff_fig5_anchor_grad_ratio.png (delta arm only, flat at 1.46,
reference line at 1.0).

Semantics note: `anchor_grad_ratio` logged by future runs is arm-faithful;
pilot curves (atyhqiir/2hepajmg) are full-graph and not comparable.

**Jeff's follow-up point (07-10):** even with stable norms, q/k/v projections
could be biased toward learning anchor-row (key-dense) features over SWA-row
features — a directional bias a norm ratio can't rule out. No measurement
exists for this yet. He also noted the stop-grad arm can still learn to "deal
with" the delta's presence even without gradients through it (forward-pass
adaptation) — see §5c for the measurement that later confirmed this.

---

## 4. LongBench evaluation (Jeff's ask, 07-10: "take those models and eval
them on different benchmarks... longbench or infinite-bench, short-generation
subtasks")

New harness eval/longbench_eval.py. Four arms sharing identical prompts:

- `base_dense` — flash_attention_2, mode=none (dense ceiling)
- `base_delta` — window/delta γ=64, no adapter
- `ce_delta` — window/delta γ=64, CE-pilot delta adapter merged
- `dense_delta` — window/delta γ=64, CE-pilot dense adapter merged (control:
  separates "PG19 LoRA of any kind" from "training through the pipeline")

Box: 1×H100 80GB (927ebf6f), 2026-07-12. Chain: 5-sample smoke on base_delta
passed first (hotpotqa sample 1: pred 'Charles L. Clifford' vs gold
['Charles L. Clifford'], F1 1.00).

### 4a. LongBench v1 QA — generation + F1 (run yt73vch2; smoke zsm4br2q)

Subtasks chosen for short answers AND contexts safely past sink+window=3072:
hotpotqa, 2wikimqa, musique (max_new_tokens 32), multifieldqa_en (64).
qasper was considered and dropped (avg ~4K tokens, too many samples under
3072 where delta ≡ dense). LongBench official prompt templates; middle
truncation to 31,500 tokens (head+tail halves, LongBench's scheme); chat
template applied; first line of generation scored; LongBench qa_f1_score
(normalize → token-level F1, max over gold answers). 50 samples/task, dataset
order (deterministic).

| arm | hotpotqa | 2wikimqa | musique | multifieldqa_en | mean |
|---|---|---|---|---|---|
| base_dense | 0.4947 | 0.4769 | 0.3017 | 0.6039 | 0.4693 |
| base_delta | 0.4541 | 0.4457 | 0.3074 | 0.6086 | 0.4540 |
| ce_delta | 0.5015 | 0.4359 | 0.2816 | 0.5856 | 0.4512 |
| dense_delta | 0.4808 | 0.4385 | 0.3414 | 0.5899 | 0.4627 |

Observations:
1. The 8K-perplexity training null replicates on QA: ce_delta ≈ base_delta ≈
   dense_delta (means within 0.012; per-task differences are non-systematic —
   ce_delta wins hotpotqa, loses musique).
2. Delta pipeline cost on LongBench QA is small: −1.5 F1 points vs dense
   (0.454 vs 0.469), far milder than the RULER gap at the same context length
   (delta γ64 73.3 vs dense 85.8 previously) — QA over long documents depends
   less on exact needle retrieval than RULER does.
3. The RULER finetune regression (−5–7 pts) did NOT replicate on LongBench QA
   (flat) — the finetune's damage is needle-retrieval-specific, not general
   long-context QA capability.

### 4b. LongBench v2 MCQ — letter-logprob scoring (run gzwk6x9u)

Rationale: Jeff trusts argmax(logprob(A/B/C/D)) scoring (his MMLU suggestion),
but MMLU prompts sit under sink+window=3072 where delta ≡ dense by
construction, so it cannot measure delta adaptation. LongBench v2 is MCQ at
long context — same low-noise scoring with the delta path active.

Method: v2 'short'-length split, filtered to prompts ≤31,500 tokens (fit
without truncation), first N in dataset order. Requested n=200; **115 samples
survived the token filter** (v2 'short' = <32K *words*, many exceed 31.5K
tokens). One prefill forward per sample, no generation; score = argmax over
logprobs of the four letter tokens at the final position; per-difficulty
split reported.

| arm | overall acc | easy | hard |
|---|---|---|---|
| base_dense | 0.3043 | 0.3810 | 0.2603 |
| base_delta | 0.2957 | 0.3095 | 0.2877 |
| ce_delta | 0.2696 | 0.2381 | 0.2877 |
| dense_delta | 0.2609 | 0.2381 | 0.2740 |

Observation: all arms near the 25% chance floor (published Llama-3.1-8B
results on LongBench v2 without chain-of-thought are similar). The instrument
has almost no headroom to separate arms at n=115 — the benchmark floor, not
the scoring method, is the limitation. Both finetuned arms drop equally
(again no delta-specific difference). Not a useful arena for this comparison;
reported for completeness.

---

## 5. 32K-context perplexity — the reversal (2026-07-12)

### 5a. First signal (8 chunks, run qlm284j1, box 927ebf6f)

Same ppl_eval method as §2a but 8 chunks @ 32,768 tokens, arms
base/delta/dense:

| arm | loss | ppl |
|---|---|---|
| base | 2.7206 | 15.189 |
| delta-trained | 2.5573 | 12.901 |
| dense-trained | 2.5776 | 13.166 |

First-ever gap between delta and dense arms (0.27 ppl), but n=8 → flagged as
needing confirmation before reporting.

### 5b. Confirmation (32 paired chunks, 4 arms, run 0reijul3, box 87047ec4)

32 PG19-test chunks @ 32,768 tokens (≤2 chunks/doc), **identical chunks
across all four arms**, per-chunk losses logged, paired differences computed.
(Different chunk set from 5a — note base ppl differs, 12.22 vs 15.19,
because the 8-chunk draw happened to include harder text; the *within-run*
arm comparison is the meaningful quantity, which is why pairing matters.)

Aggregate:

| arm | loss | ppl |
|---|---|---|
| base | 2.5032 | 12.222 |
| delta-trained | 2.3392 | 10.373 |
| detach-trained | 2.3484 | 10.469 |
| dense-trained | 2.3594 | 10.585 |

Paired per-chunk loss differences (mean ± sem over 32 chunks):

| pair | mean diff | sem |
|---|---|---|
| base − delta | +0.1640 | 0.0043 |
| base − dense | +0.1438 | 0.0038 |
| base − detach | +0.1548 | 0.0041 |
| **delta − dense** | **−0.0202** | **0.0011** |
| delta − detach | −0.0092 | 0.0005 |
| dense − detach | +0.0110 | 0.0008 |

Per-chunk raw losses (chunk order identical across arms):

```
base:   2.1051 3.2000 3.0368 2.4500 2.6543 2.9728 2.9491 2.3966 2.3391 2.2672
        2.5941 2.5652 2.5942 2.5100 2.4022 2.4302 2.5488 2.4763 2.2762 2.3345
        2.4644 2.4620 2.5724 2.4760 2.4025 2.3033 1.9481 2.5505 2.5502 2.4067
        2.4572 2.4073
delta:  1.9560 2.9956 2.8850 2.3182 2.5171 2.8015 2.8020 2.1830 2.1455 2.1102
        2.3976 2.4031 2.4122 2.3630 2.2299 2.2794 2.3501 2.2918 2.1297 2.2170
        2.2686 2.3015 2.3836 2.3276 2.2303 2.1800 1.7908 2.3881 2.4133 2.2416
        2.2887 2.2530
dense:  1.9698 3.0277 2.9044 2.3338 2.5436 2.8153 2.8174 2.2090 2.1752 2.1286
        2.4159 2.4210 2.4254 2.3791 2.2684 2.3086 2.3742 2.3163 2.1418 2.2303
        2.2920 2.3211 2.4107 2.3415 2.2472 2.1937 1.8094 2.4048 2.4298 2.2660
        2.3053 2.2751
detach: 1.9613 3.0085 2.8934 2.3267 2.5292 2.8086 2.8087 2.1939 2.1580 2.1183
        2.4073 2.4146 2.4210 2.3731 2.2446 2.2925 2.3604 2.3022 2.1363 2.2220
        2.2795 2.3083 2.3931 2.3323 2.2374 2.1856 1.8008 2.3956 2.4208 2.2542
        2.2972 2.2642
```

### 5c. Observations

1. **The delta-trained model beats the dense-trained control on all 32 of 32
   chunks** (paired −0.0202 ± 0.0011, ≈18 sem from zero). The 8K triple-null
   does not hold at 32K.
2. **Monotone ordering in gradient flow through the delta:** delta (full
   gradients) < detach (forward-only exposure) < dense (no pipeline at all),
   with each pairwise gap individually significant. A dose-response
   relationship: more gradient through the delta path → better long-context
   pipeline perplexity.
3. **Mechanistic reading of the 8K null:** at seq len 8192 with sink+window
   = 3072, only ~60% of rows are past the sparse boundary (and only ~77
   anchors/sequence); at 32,768, ~90% of rows are. The training pressure on
   the delta path exists but is diluted at 8K below detectability at n=16
   (8K arm differences ~0.002 ppl ≈ replicate noise); at 32K the same
   adapters separate cleanly. The adapters were trained at 8K, so the effect
   also generalizes beyond the training length.
4. **Jeff's stop-grad remark is confirmed quantitatively:** the detach arm
   (delta present in the forward, no gradient through the correction)
   captures roughly half the delta arm's advantage over dense
   (−0.011 of −0.020) — the network does learn to "deal with the information
   in the delta" without gradients through it, but full gradients through
   the correction add a further, separately significant, improvement.
5. Effect size honesty: the delta-vs-dense gap is ~0.85% of loss / 0.21 ppl.
   Real and reproducible, not large. Levers that plausibly scale it, untested:
   training at 32K (4× the delta-active loss fraction), distillation loss
   (put the objective directly on the correction error), longer training.

---

## 6. Corrections and caveats affecting previously-communicated claims

1. **Retracted (never sent):** "the gradient-stopped arm shows the same 1.46×
   ratio, so the γ-fold accumulation doesn't contribute" — instrumentation
   artifact (§3); the probe ignored the detach flag. Caught before sending;
   Slack message was corrected to remove the claim.
2. **anchor_grad_ratio precision:** 1.46× is query-gradients at layer 0
   specifically, not "q/k/v projections" generally.
3. **RULER subset framing:** every RULER number in this record is the fixed
   3-task subset at 32K, 150 prompts per number — internally consistent, not
   comparable to published full-RULER averages. Jeff (07-10): "That is enough
   I think."
4. **Observed run-to-run noise** (from exact duplicate chains): RULER ≤1.4
   points on 150 prompts; 8K ppl ~0.002; decode numbers replicated exactly.
5. **8-chunk vs 32-chunk 32K base ppl differ (15.19 vs 12.22)** — different
   chunk draws; cross-run ppl levels are not comparable, within-run paired
   arm differences are the reliable quantity.
6. **v2 MCQ n=115 not 200** — the token-budget filter, see §4b.

## 7. Where everything lives

- wandb project `singhh5050-stanford-university/delta-attention`; run IDs
  inline above; box-final-state artifacts: xfxl13hv (8cb748cf), fxtkk4gp
  (25dad994), uuy9z2f5 (cb303dba), sfj54cz2 (927ebf6f), gswww95x (87047ec4).
- Adapters: wandb artifacts `wp2_adapter_{delta,dense,detach}:latest`.
- Code (all on master, github.com/singhh5050/delta-attention):
  eval/ppl_eval.py (per-chunk + paired stats), eval/longbench_eval.py,
  eval/run_wp.sh modes t2eval/longbench/ppl32k,
  delta_attention/train/train_delta.py (arm-faithful probe + decomposition),
  tests/test_delta_decode.py (per-layer/refresh verification),
  tests/test_longbench_offline.py.
- Figures: figures/jeff_fig5_anchor_grad_ratio.png (plus the four 07-09
  update figures).

## 8. Open items / next experiments (not yet run)

- Distillation-loss pilot (~$20): corrected outputs vs frozen dense teacher —
  the objective that targets the correction error directly; also the untested
  half of Jeff's "decode needs the finetuned model" idea.
- Training at 32K (~$25): moves the training signal to where the effect lives.
- anchor_grad_ratio decomposition (free — logged automatically by the next
  delta-arm training run): does the γ-fold summed correction contribute
  meaningfully to the 1.46, i.e. is a 1/γ backward scaling targeting anything?
- Directional q/k/v bias probe (Jeff's refined concern): no measurement
  designed yet.
- Prefill cached-vs-true delta curve at ν=1..63 (the fig-2 companion line):
  probe modification sketched, not run.
- InfiniteBench En.MC at ~100K+ (logprob-scorable): deferred, slow per sample.
