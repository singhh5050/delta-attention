# STATS — the single results ledger

Every headline number in this project, with its wandb run ID. Raw CSVs
are triple-archived (wandb run + box_final_state artifact + rescue/).
Conventions: raw numbers only; paired = per-chunk/per-sample diffs on
identical inputs, mean ± sem. Quick reference: PROJECT_CHEATSHEET.md;
how to rerun anything: ../REPRODUCING.md.

Two parts:
- **Part I — current track**: training-efficiency (delta as a cheaper
  long-context training path) and MTP/speculative-decoding (delta reads
  inside production drafter heads). Sections §O–§V2.
- **Part II — pre-pivot archive**: self-speculative decoding,
  distillation, seeds, gradient probes on Llama-3.1-8B. Superseded
  framings but the provenance for numbers already shared externally.

---

# Part I — current track


## O. Paper-core triad (07-17, box 30, master@c8b89e4; raw:
rescue/2026-07-17-triad/ + wandb box_final_state:v20)

### O4. T1 — DEFINITIVE training-efficiency numbers (07-20 clean rerun,
box 32; supersedes the O1 table below)

Idle 1×H100 (nothing co-located), strictly sequential, GPU-health
preflight PASSED (burn-in then 1980/1980MHz @ ≤60C; clocks+temp now
recorded per row). Probe-free peak memory. Both dense kernels benched.
Raw: rescue/2026-07-20-trainbench.csv. NOTE: the FIRST rerun attempt
(box 31) had a thermally throttled GPU — 495/1980MHz @87C, all numbers
2.4× slow — terminated before archiving; preflight gate added
(run_wp.sh) so this failure mode is now structural, not luck.

| arm | impl | seq | fwd ms | bwd ms | step ms | tok/s | peak GB |
|---|---|---|---|---|---|---|---|
| delta | flex | 8K | 352 | 905 | 1263 | 6484 | 20.36 |
| detach | flex | 8K | 353 | 897 | 1257 | 6519 | 20.36 |
| dense | sdpa | 8K | 359 | 845 | 1211 | 6766 | 20.36 |
| dense | fa2 | 8K | 350 | 822 | 1178 | 6954 | 20.36 |
| delta | flex | 32K | 1451 | 4560 | 6019 | 5445 | 33.00 |
| detach | flex | 32K | 1451 | 4539 | 5997 | 5464 | 33.00 |
| dense | sdpa | 32K | 2034 | 5653 | 7695 | 4258 | 32.95 |
| dense | fa2 | 32K | 1929 | 5420 | 7357 | 4454 | 32.58 |

**Headline (vs the FAIR fa2 dense baseline): delta trains 1.22× faster
per step @32K (fwd 1.33×, +22% tok/s) and 7% SLOWER @8K.** Against sdpa
the ratios are 1.28×/1.40× — the kernel confound was real and worth
~5 points of speedup; quote the fa2 numbers.

- The 07-17 concurrent-load concern did NOT materialize: box-30 numbers
  reproduce within ~1% (delta 6084→6019, dense-sdpa 7757→7695). The
  caveat was correct process; the data survived it.
- Memory: peaks identical across arms @8K (20.36 — structure-dominated);
  @32K delta 33.00 vs fa2-dense 32.58 (+1.3%). "Approximately equal,
  delta marginally higher" — not "equal".
- Dense arms downclock mildly under load (1680–1740MHz vs delta's
  1935+) — normal DVFS at higher compute intensity, part of real dense
  training cost, included in the measurement.
- GPU-hours for a 500-step 32K run: delta 0.84h vs dense-fa2 1.02h.

### O1. T1 — SUPERSEDED first measurement (07-17; kept for provenance only — quote §O4, not this table)

**VALIDITY CAVEATS (07-20 review) — clean rerun in §O4 supersedes these
numbers:** (1) this grid was timed CONCURRENTLY with T3's 32K training on
the other GPU of the same host (shared CPU tokenization/PCIe/host
bandwidth; delta/detach step sems 3–10× dense's, consistent with
interference) — direction @32K is too large to be contention, but
absolute numbers and the marginal 8K result are suspect; (2) the dense
arm ran under HF `sdpa` while every eval-side dense path uses
flash_attention_2 — kernel choice vs mechanism unvalidated; (3) peak
memory includes final-step probes that run delta-pipeline math for every
arm (identical 20.36GB across arms at 8K is the tell) — "equal memory"
NOT demonstrated by this data.

CUDA-synced fwd/bwd/step over 30 timed steps after 5 warmup, identical
LoRA/loop across arms (runs przlkl22/7gpkp7fy/vbiq24yy/rk2tyleu/n4i01dgf/
gpspcibr; smoke p4civrv9):

| arm | seq | fwd ms | bwd ms | step ms | tok/s | peak GB |
|---|---|---|---|---|---|---|
| delta | 8K | 356 | 911 | 1274 | 6429 | 20.4 |
| dense | 8K | 364 | 856 | 1226 | 6682 | 20.4 |
| detach | 8K | 357 | 905 | 1268 | 6459 | 20.4 |
| delta | 32K | 1471 | 4606 | 6084 | 5386 | 33.0 |
| dense | 32K | 2054 | 5696 | 7757 | 4224 | 32.9 |
| detach | 32K | 1471 | 4581 | 6060 | 5408 | 33.0 |

- **@32K: delta trains 1.27× faster per step (fwd 1.40×, bwd 1.24×),
  +27.5% tokens/sec** — subject to caveats (1)+(2) above; see §O4.
- **@8K: delta measured ~4% SLOWER** (a 48ms gap with sems up to 7.4ms
  under co-located load — could move on an idle box; see §O4). Mechanism
  prior: window covers 37% of context @8K, so little attention to save.

### O2. T2 — downstream retention, made paired (run ivi83fyq force-dense;
18brahqv En.MC)

LongBench v1 QA, all arms FORCE-DENSE eval (capability retention of
trained weights), n=50/task, paired per-sample:

- **delta-trained − dense-trained: −0.014 ± 0.022 (n=200, 21W/24L/155T).**
  HONEST READING (07-20 review): no difference DETECTED, but the 95% CI is
  roughly (−0.058, +0.030) F1 — this n rules out large regressions only; a
  4–5-point drop is compatible with the data. "Parity" in the equivalence
  sense would need ~4× the n or an explicit TOST margin. delta-trained −
  base: −0.018 ± 0.028; dense-trained − base: −0.004 ± 0.023.
- En.MC (pipeline eval, n=229, first run for these arms): ce32k 0.5415,
  dense32k 0.5677, dftmix 0.5459 — spread ~2.6pts vs mcq sem ≈ 3.3pts:
  within noise.

### O3. T3 — second-corpus replication (arXiv, common-pile parquet,
cross-doc packed; 500 steps @32K, ~16.4M tokens, same recipe as PG19)

**PROTOCOL CAVEATS (07-20 review):** (a) "held-out, disjoint by
construction" was WRONG as stated — streaming shuffle() permutes corpus
shards, so training read shard-10 docs (canonically ~140K deep) while
eval read unshuffled docs 20K+: disjoint FOR THIS RUN (verified post-hoc
via the shard permutation), by luck not construction. Loader now pins
training to take(15000) before shuffle → structural for future runs.
(b) The 32 eval chunks were packed back-to-back from ~100 consecutive
docs — neighboring chunks share documents, so the ±sems below treat
autocorrelated chunks as independent and overstate significance;
protocol fixed (one chunk per doc-window, 20-doc spacing) for reruns.
(c) arXiv differs from PG19 in corpus AND packing (cross-doc eos-packed,
loss computed across paper boundaries, vs per-doc book chunks) AND eval
provenance (same-split held-out vs separate test split) — three variables
moved at once.

Pipeline eval (loss, 32 paired chunks): base 0.8799, delta-ft 0.7851,
dense-ft 0.8303. **Paired delta − dense = −0.0452 ± 0.0079 — delta-trained
better under the pipeline, REPLICATING PG19 (−0.0248) with a LARGER
margin.** Dense eval: base 0.7245, delta-ft 0.7065, dense-ft 0.6903;
paired delta − dense = +0.0162 ± 0.0013 (dense-ft better under dense
eval, replicating PG19's +0.0173 almost exactly).

- Retained fraction of the dense-finetune benefit under dense eval:
  0.0180/0.0342 = **53% on arXiv vs 88% on PG19** — the qualitative
  pattern replicates, but per caveat (c) DO NOT attribute the fraction
  difference to corpus: packing and eval provenance are confounded with
  it. (Scale note: arXiv base losses are 2.8–3.3× smaller than PG19's in
  ln units, so both retention denominators are small. Several arXiv eval
  chunks have near-zero base loss (0.12–0.20 dense) — likely
  highly-predictable/boilerplate LaTeX spans.)
- Pipeline tax, CORRECTED (07-20 review — the original bullet compared
  incompatible units and was INVERTED): in A3's ln-units, arXiv base tax
  = 0.8799 − 0.7245 = **0.1554 vs PG19's 0.0815 — the tax is ~1.9×
  LARGER on arXiv**, not smaller. The earlier "shorter long-range deps →
  smaller tax" mechanism claim is RETRACTED; the data suggests the sparse
  approximation hurts arXiv MORE (hypothesis, unmeasured: LaTeX
  cross-references/notation dependencies exceed the 2K window), even
  though delta-training also recovers more there (−0.045 paired).

Triad status: T1 direction measured, absolutes pending the idle-box
rerun (§O4); T2 = no detected difference at limited power (NOT
equivalence-grade parity); T3 = paired directions replicate PG19 on both
evals, with protocol caveats (a)–(c). Box self-terminated; cost ~$20.


## P. MTP Track A — Phase A mechanism probe (07-21, box 34; raw:
rescue/2026-07-21-mtpa/ + wandb box archive; harness reviewed twice,
commits ffa17a5 + 6472ce2)

Setup: DeepSeek/MCore-style 1-layer MTP module (shared emb + proj + one
transformer block + shared head, warm-started from trunk layer 31) on a
FROZEN dense Llama-3.1-8B trunk; module attention is the ONLY variable
(dense vs delta = pipeline prefill + anchor-corrected decode at gamma=64).
Trained 2000 steps x 8K PG19-only (~16.4M tokens; the planned chat-mix was
cut for the same-day deadline) predicting t+2, teacher-forced parallel.
Final losses: dense 3.6605 (run mhhuf1pv), delta 3.7072 (p1zq1d3d) —
sensible t+2 difficulty (~1.4 nats over next-token), so the modules
trained correctly. Eval: true draft-and-verify @32K, output dense-greedy
by construction, parity full (QA) / 54+1tie (GovReport). Runs 3v5rg9i5
(gov n=10) / kxxfbkzr (qa n=20).

| module | suite | K=1 acc | K=2 pos-2 | K=4 pos-3/4 |
|---|---|---|---|---|
| dense | gov | 0.2096 | 0.0340 | ~0 |
| delta | gov | 0.1973 | 0.0293 | ~0 |
| dense | qa | 0.2314 | 0.0427 | 0 |
| delta | qa | 0.2114 | 0.0164 | 0 |

**Pre-registered Phase A gate (pos-2 >= 0.5): FAILED decisively.** A
1-layer head under THIS recipe (16M tokens, books-only, post-hoc) cannot
draft an instruct model on chat-templated long-context tasks; chaining a
depth-1-trained module (K>1) collapses immediately — consistent with why
Nemotron trains "repeated MTP" (the reuse is trained, not improvised).

**What survived — the transferable result: delta ~= dense INSIDE the
head.** Acceptance delta 1.2-2.0 points absolute (~6-9% relative) at 32K,
training-loss delta +0.047 — the delta-attention swap inside an MTP head
is nearly free, measured under a certified harness. This is the number
that matters for delta-ifying a PROPERLY-trained head (Nemotron 3 Super's,
trained at pretraining scale with published short-context acceptance
~3.45): the mechanism costs little; the head quality comes from training
scale we should not replicate ourselves.

Honest scope note: the gate as pre-registered conflated "1-layer heads
cannot draft at long context" with "a 25-minute post-hoc recipe cannot
train one" — this run demonstrates the latter. Distinguishing them would
take EAGLE-scale training (~days), which Track B/C on the production head
makes unnecessary.

## Q. Sparse-branch kernel diagnostic (swabench, 07-21, box 37 — Jeff's
"is flex the problem" test; runs vxyihvka (8K) / 7i7l54rt (32K); raw:
rescue/2026-07-21-swabench/swabench.csv + wandb box archive)

30 synced full training steps/cell after 8 warmup; variants differ ONLY in
the sparse branch (anchor/correction branches bit-identical across
variants); fa2swa rows are timing-only (no sink = knowingly wrong math).

| variant | 8K step ms | 32K step ms | vs flex @32K |
|---|---|---|---|
| delta-flex (production) | 1274.2 | 6086.9 | — |
| delta-flexgqa | 1285.1 | 6115.8 | +0.5% |
| delta-fa2swa-2048 | 1252.7 | 5916.1 | −2.8% |
| delta-fa2swa-3072 (sink-sized) | 1298.4 | 6166.1 | +1.3% |
| dense-fa2 (reference) | 1191.6 | 7428.0 | — |

**Verdict: FlexAttention exonerated.** FA2 native SWA doing LESS work (no
sink) is only 1.7–2.8% faster; sink-size-bracketed it's a wash. flexgqa ≈
flex → the GQA-expanded sparse read isn't it either. The gap between
delta's measured fwd (1470ms @32K) and its theoretical attention floor
lives in the SHARED branches — anchor-row masked-SDPA + correction
broadcast — which is where kernel work should aim. Caveats: 2-GPU box (one
used, sequential); temp/clock CSV columns mangled by a 2-line nvidia-smi
parse (timing columns unaffected); fa2swa-3072 mildly downclocked
(1665MHz) — noise at these margins.

## R. MiMo-7B-RL production MTP head × delta (07-21, box 36; runs
rvyz2m3o/+; raw: rescue/2026-07-21-mimo/ + wandb box archive v23; harness
per docs/mimo_mtp_plan.md, all parity gates green)

M0 calibration: K=1 acceptance 0.7136 @3K chat QA (Xiaomi claims ~0.9 on
their reasoning evals — domain difference; wiring certified by parity).

K=1 acceptance, dense head vs ZERO-SHOT delta swap (no retraining;
prompt_tok_mean = measured mean prompt length — documents cap near
10–12K, so the top two tiers are near-duplicates and the measured column
is the honest x-axis):

| suite | measured len | dense | delta (0-shot) | Δ |
|---|---|---|---|---|
| qa | 3996 | 0.7058 | 0.7133 | +0.8 |
| qa | 7674 | 0.6938 | 0.6976 | +0.4 |
| qa | 12010 | 0.6769 | 0.6595 | −1.7 |
| qa | 12224 | 0.6645 | 0.6652 | +0.1 |
| gov | 4115 | 0.5882 | 0.5813 | −0.7 |
| gov | 7407 | 0.5732 | 0.5743 | +0.1 |
| gov | 10677 | 0.5549 | 0.5608 | +0.6 |
| gov | 10722 | 0.5432 | 0.5556 | +1.2 |

- **Headline: the zero-shot delta swap on a production-trained MTP head
  is FREE within noise** — |Δ| ≤ 1.7pts across all 8 cells, sign flips
  both ways, mean Δ ≈ +0.1pt. No retraining (M3 may be unnecessary for
  the core claim).
- **Acceptance vs context (dense head): gentle decay, no collapse** —
  qa 0.714→0.665, gov 0.588→0.543 over ~3× length growth (~5pts). BUT
  measured only to ~12K real tokens: LongBench documents are too short
  to populate the 16K/31.5K tiers (the review-mandated prompt_tok_mean
  column is what exposes this). Extending the curve to real 32K needs
  long documents (InfiniteBench books) — queued as a follow-up cell.
- Chaining (K=2/4, exploratory): collapses identically for BOTH variants
  (pos-2 ≈ 0.03) — the head is depth-1-trained and vLLM's K=1-only
  support is vindicated; delta ≈ dense even in the collapse.

## S. Model-2 replication: Qwen3-14B training-efficiency 2x2 (07-22, box 33;
## runs vwnp2zf6 pipeline-eval, wz35nqkp dense-eval; trainings m2-train-delta /
## m2-train-dense on the same box, archived in box_final_state @ box 33)

Setup: Qwen3-14B (2025 release, QK-norm port, portgate PASSED vs vanilla
transformers), LoRA r16 q/k/v/o, 500 steps @ 32K PG19, delta vs dense arms
trained SEQUENTIALLY on one idle H100 (O4 discipline). Eval: 32 held-out
PG19 TEST chunks @ 32768, identical chunks across arms, chunked fused-CE
(the labels= path materializes seq x 152K-vocab fp32 logits = ~19GB and
OOM'd the first eval attempt, run n282nw74 — fix dccd103, adapters were
safe on disk; both paid trainings unaffected). All losses nats/token under
the QWEN tokenizer — NOT comparable to any Llama number in this file.

| eval path      | base   | delta-ft | dense-ft | paired delta-dense (sem) | chunks |
|----------------|--------|----------|----------|--------------------------|--------|
| pipeline (g64) | 13.291 | 10.690   | 11.427   | -0.0666 (0.0018)         | 32/32 delta better |
| dense (fa2)    | 11.533 | 10.303   | 10.065   | +0.0234 (0.0014)         | 32/32 dense better |

- The Llama-3.1-8B pattern REPLICATES on a second, newer architecture:
  each arm wins under its own eval path, and delta's advantage under the
  pipeline (0.067 nats) is ~3x dense's advantage under dense (0.023) —
  the same asymmetry as Llama's 32K 2x2 (§ stats_2026-07-15.md).
- Capability retention: delta-ft under DENSE eval (10.30) recovers most
  of the dense-ft gain over base (11.53 -> 10.06), i.e. delta training
  does not wreck the model's native-attention quality.
- Every paired comparison is 32/32 unanimous across chunks — not a mean
  driven by outliers.

## T. Anchor-branch decomposition + Jeff's 1M ladder (anchorbench, 07-22,
## box 39; short ladder run zofiwhdq @ f4da7c5+dccd103, CSV rescued to
## rescue/2026-07-22-anchorbench/; long ladder completed across §T2/§T3)

Jeff's protocol verbatim: weightless single attention layer, torch.randn
inputs, fwd/bwd, Llama-8B dims (32 q / 8 kv heads, D=128), gamma 64,
sink 1024 + window 2048. ALL numbers PER LAYER.

Short ladder (8K / 32K), fwd+bwd ms unless noted:

| cell               | 8K    | 32K   | reading |
|--------------------|-------|-------|---------|
| sparse-flex        | 3.69  | 16.73 | production sparse branch |
| anchor-masked      | 4.53  | 43.36 | production anchor branch — DOMINATES at 32K |
| anchor-flash!      | RAISED| RAISED| "No available kernel": the row mask provably knocks SDPA off the flash path (the smoking gun) |
| anchor-mathonly    | 5.32  | 44.76 | forced math backend |
| anchor-efficient   | 4.53  | 43.32 | == anchor-masked: SDPA silently picks mem-efficient, whose BACKWARD is the real cost |
| anchor-flexrow     | 1.06  | 5.26  | reformulation: 8.2x faster than production anchor branch at 32K |
| correction         | 0.58  | 1.98  | trivial, as designed |
| full-delta-current | 9.16  | 63.12 | production delta_forward_train verbatim |
| full-delta-flexrow | 5.67  | 25.08 | same math, flexrow anchor branch |
| full-dense         | 6.70  | 95.20 | causal SDPA reference |

- Verdict on Jeff's question: NOT flex (swabench §Q already exonerated
  it); the anchor branch's masked SDPA falls to the mem-efficient
  backend and its backward is ~2/3 of the whole per-layer cost at 32K
  (43.4 of 63.1ms).
- full-delta-current beats dense 1.5x at 32K but LOSES at 8K (9.2 vs
  6.7) — the attention-layer view of O4's crossover. The flexrow
  composition wins at BOTH lengths (5.67 vs 6.70 at 8K; 3.8x dense at
  32K) — the "way faster" Jeff suspected exists, without any Triton
  authoring, and it is a drop-in reformulation of the anchor branch.
- headread cells (MiMo "is the head cheaper" question): dense one-token
  read 0.06ms at 32K cache vs delta amortized ((63*0.07+0.06)/64 —
  window-read dominated) — per-layer read cost is negligible at 32K;
  the curve to 1M (where it stops being negligible) is in the long
  ladder.
- Long-ladder incident log (all recorded as data, fixes pushed):
  (1) eager create_block_mask materializes the full s^2 bool mask —
  128GB block-sum intermediate at 131K; fixed with _compile=True
  (785c837), which also UN-CAPS the production composition: it RUNS at
  262K (fwd+bwd 748.25ms vs flexrow 292.70ms, run nkz99x1e summary).
  (2) flex triton templates are int32-indexed (torch 2.8): >=524K with
  32 heads exceeds 2^31 elements per tensor -> RAISED-64BIT recorded as
  the production formulation's per-GPU cap; head-chunked -hc4 cells
  (1895aad) carry the delta curve to 1M.

### T2. Long ladder DEFINITIVE (07-22, box 39, run n6qht8yh @ 1895aad;
### CSV + logs: rescue/2026-07-22-anchorbench/final/ + box_final_state
### artifact @ run rys676u0)

Whole-function fwd+bwd, ms PER LAYER (multiply by n_layers for a step):

| cell                     | 131K        | 262K        | 524K         | 1M   |
|--------------------------|-------------|-------------|--------------|------|
| full-dense (causal SDPA) | 1523.6      | 6110.8      | 24432.4      | OOM  |
| full-delta-current       | 328.3 (4.6x)| 750.3 (8.1x)| int32-cap    | OOM  |
| full-delta-flexrow(-hc4) | 122.8 (12.4x)| 293.1 (20.9x)| 920.9 (26.5x)| OOM |

- The delta training-efficiency story SCALES: the 32K step-level 1.22x
  (O4) grows to 4.6x/8.1x/26.5x at the attention layer as context grows —
  exactly the long-context-effect framing, now with Jeff's own protocol.
- 524K: production composition hits the torch-2.8 flex int32 cap
  (RAISED-64BIT recorded); the -hc4 head-chunked flexrow carries it.
- 1M: NOTHING whole-function fits on one 80GB H100 with backward — dense
  included, so no comparison point exists on this hardware. Components
  that DO run at 1M: anchor-flexrow 1747.2, correction 57.1, gqa-expand
  16.5 (sparse-flex-hc4 OOMs on its expanded 32-head K/V copies).
  NOTE anchor-flexrow @1M is allocator-MARGINAL: it ran here (n6qht8yh)
  but OOM'd in two later processes with different preceding-cell
  residue (16ja2q7c, 3y4k2c44) — treat 1747.2 as "fits with a clean
  allocator", not unconditional. The composed 1M delta number landed
  via the no-grad infer cells instead (§T3).
- Anchor branch confirmed as the scaling bottleneck of the CURRENT impl:
  anchor-masked fwd+bwd 247 -> 585 -> 2097ms (131K->524K), i.e. the
  mem-efficient-backend backward, while flexrow does the same math in
  37 -> 123 -> 453ms.
- headread (the "is the MTP head cheaper with delta" answer, one-token
  read per layer): dense grows linearly 0.19 / 0.36 / 0.71 / 1.38ms
  (131K->1M); delta window-read is FLAT ~0.08ms at every length ->
  amortized ((63*delta + dense)/64) ~0.08-0.10ms. Per drafted token the
  delta read is ~2.4x cheaper at 131K and ~14x at 1M, and unlike dense
  it does not grow with cache length.

### T3. 1M closed out (07-22; sources: box-40 full ladder 16ja2q7c
### @e3ff495 [+short jcqso3pg, archive nfl5gbyz, CSV
### rescue/2026-07-22-anchorbench-v2/], box-41 v3 131K+1M run 3y4k2c44
### @0f2add5, box-41 v4 1M-delta-only run 2c3ufi39 @f6beab5 [archives
### xend1cbc/rpf5y49q, CSVs rescue/2026-07-22-box41-final/])

INFERENCE latency (no-grad fwd, ms PER LAYER; 131K/1M from dedicated
infer-* cells, 262K/524K from the fwd_ms column of the fwd+bwd cells —
validated at 131K where both exist: 430.0 vs 432.5, <1% apart):

| cell            | 131K  | 262K   | 524K   | 1M      |
|-----------------|-------|--------|--------|---------|
| dense           | 432.5 | 1761.0 | 7128.0 | 29004.6 |
| delta (current) | 63.2  | 188.2  | int32  | —       |
| delta (flexrow) | 30.3  | 78.8   | 202.6  | 612.5   |
| ratio (flexrow) | 14.3x | 22.3x  | 35.2x  | 47.4x   |

- THE 1M point Jeff asked for: 612.5ms (flexrow, 4 head chunks) and
  600.8ms (lowmem in-place variant — independent implementation, same
  math, 2% apart = mutual confirmation) vs 29.0s dense.
- Repro stability: 1M infer-dense measured in two independent processes
  — 29004.6 (16ja2q7c) vs 29151.1 (3y4k2c44), 0.5% apart; 131K infer
  cells likewise (dense 428.2/432.5, delta-current 63.1/63.2, flexrow
  30.1/30.3). The 262K/524K fwd_ms proxy is validated against TRUE
  no-grad infer-dense from 16ja2q7c: 1761.0 vs 1770.9 (262K) and 7128.0
  vs 7186.0 (524K), both <1%. No true no-grad DELTA measurement exists
  at 262K/524K (pre-fix runs OOM'd those cells) — only those two delta
  entries in the table above are proxies.
- Training (fwd+bwd) at 1M remains unmeasurable on one 80GB H100 for the
  delta composition (full-delta-chunkbwd4 OOM even in a minimal-residue
  process, run 2c3ufi39); dense needed chunked backward (98.6s). The
  training curve stands on 131K/262K/524K (12.4x/20.9x/26.5x, §T2).
- Caveat: the v4 process ran no length below the int32 cap, so its hc
  parity cert is marked UNCERTIFIED in-process (first CSV row, by
  design); flex_hc parity was certified at maxdiff EXACTLY 0.0 in three
  other processes across two boxes (8K x2, 131K).
- Incident ledger for the 1M chase (all recorded as rows): eager block
  mask O(s^2) OOM -> _compile=True; flex int32 cap -> head chunking;
  anchor-masked mem-efficient backward ILLEGAL MEMORY ACCESS under
  expandable_segments (poisons CUDA context) -> --skip with exact-match
  labels + KNOWN_CELLS validation; residue allocations (correction
  leaves, chunkbwd seed) gated behind skips.

## U. Gemma 4 native MTP drafter — G0 baseline (07-22, box 41; v2 runs
## 1utfcnzp smoke / qobgwn2z tiers (superseded methodology), v3 runs
## ygdlc85d smoke / vx6ldm79 tiers (two-length differencing + symmetric
## DynamicCache); run_ids verified against the CSV rows; CSVs in
## rescue/2026-07-22-box41-final/ + archives v3ih9yss/xend1cbc/rpf5y49q)

- UPSTREAM BUG (transformers 5.15.0.dev0): assisted decoding with any
  prompt past the trunk sliding window (1024) crashes in the verify
  forward (hybrid cache window-caps sliding KV, mask built full-length;
  reproduces under sdpa AND eager). Workaround: explicit DynamicCache,
  BOTH modes, parity-gated.
- G0 gate PASS (fraction gate; parity scatter = kernel tie-flips in the
  batched verify path, pp spread 6..256 with ~40% exact rows).
- Decode-only speedup (two-length differencing, 256 max_new, PG19,
  n=10/tier, run vx6ldm79 — the corrected methodology):

  | tier | plain tok/s | assisted tok/s | speedup median [min-max] | exact | pp median |
  |------|-------------|----------------|--------------------------|-------|-----------|
  | 4096 | 13.8        | 24.0           | 1.43 [1.18-4.57]         | 1/10  | 84.5      |
  | 8192 | 9.9         | 13.5           | 1.36 [1.08-1.64]         | 1/10  | 74.0      |

  QUOTE THE MEDIANS: the 4K mean (1.73x) is dragged by one 4.57x
  fully-accepted outlier prompt. Below Google's ~3x claim — ours is the
  naive HF assisted path, single prompt, no server batching/CUDA graphs.
  (v2-methodology 4K rows from run qobgwn2z are superseded — asymmetric
  caches + prefill conflation — but agree in ballpark, 1.16-1.29x.)
- Tiers >= 16K OOM on 80GB in EVERY mode (both cache types; run
  qobgwn2z, all 30 rows at 16K/32K/64K) — consistent with full-sequence
  prefill logits (16K x 262K vocab); 8K fits (vx6ldm79). G1's
  custom loop prefills with logits_to_keep=1 and owns the draft loop,
  which removes both this cap and the upstream-bug workaround.
- KNOWN CAVEAT on v3 speedups: the heuristic num_assistant_tokens
  schedule was NOT reset between the half and full generates (fix
  2c34534 landed after the box terminated), so each speedup mixes two
  speculation depths. Direction/magnitude small (walls are consistent
  across rows); G1 re-measures under our own loop. Do not quote v3
  speedups beyond "~1.2-1.5x ballpark".

## V. Gemma 4 G1 — delta-reading the native MTP drafter (07-22, boxes
## 42-44; CERTIFIED run 92y2luja @711817f, CSVs
## rescue/2026-07-22-g1-v3/ + archive ogv8a0ra; earlier iterations
## f9juentk/zmqfnu2f superseded — see caveats)

Design: our own draft-verify loop (transcribed verbatim from
transformers' Gemma4 candidate generator; eval/gemma4_g1_eval.py). The
drafter is NEVER modified — arms differ only in the shared_kv_states
view it is handed: full / sparse (full_attention entry subsampled to
sink1024+window2048) / deltaN (full read every Nth drafter call). The
trunk verifies greedily, so acceptance is exact and outputs are the
trunk-greedy chain BY CONSTRUCTION (output "quality" cannot differ
across arms; acceptance is the metric).

Gates (all green on 92y2luja): zero-tolerance parity vs plain trunk
greedy computed with IDENTICAL forward shapes (64/64 x4 at 4K+8K);
independent short-context native-assisted cross-check 32/32.

| tier | arm    | acc/round | acc_rate | draft ms | n |
|------|--------|-----------|----------|----------|---|
| 4096 | full   | 1.97      | 0.40     | 4.4      | 6 |
| 4096 | sparse | 1.95      | 0.39     | 4.2      | 6 |
| 4096 | delta2 | 1.96      | 0.39     | 4.3      | 6 |
| 4096 | delta4 | 1.96      | 0.39     | 4.2      | 6 |
| 8192 | full   | 1.18      | 0.24     | 5.0      | 6 |
| 8192 | sparse | 1.19      | 0.24     | 4.2      | 6 |
| 8192 | delta2 | 1.20      | 0.24     | 4.6      | 6 |
| 8192 | delta4 | 1.18      | 0.24     | 4.4      | 6 |

- **HEADLINE (as far as measured): the sparse read of the trunk cache is
  FREE for the trained multi-step drafter** — at 8K only 37% of the
  cache is visible to the sparse arm and acceptance is indistinguishable
  from the full read (1.19 vs 1.18/round), while the drafter call is
  ~16% cheaper (4.2 vs 5.0ms; gap grows with cache length).
- The earlier apparent 8K gap (sparse 1.01 vs full 1.08, box-43 run
  zmqfnu2f) VANISHED under corrected sliding-entry semantics — it was an
  artifact of handing the drafter full-length sliding states (uncapped
  workaround cache), not a delta effect. Boxes 42/43 absolute numbers
  are superseded; their arm-deltas were directionally fine.
- Acceptance drops with context for ALL arms including full (1.97 ->
  1.18/round from 4K to 8K): a property of the native drafter on long
  PG19 continuations, not of delta-ification.
- >=16K blocked twice: uncapped cache = 62GB weights + 16.6GB KV @32K
  (box 43); native hybrid cache still 76GB ALLOCATED at 16K prefill
  (box 44 memory dump — prefill transients + allocator litter). v4
  (in flight, box 45): CPU-offloaded uncapped cache + drafter sliding
  entry capped to native window+1 by us + inter-arm cleanup; offload
  slows decode but cannot affect acceptance.
- Parity above 8K is structurally unverifiable against native (native
  assisted OOMs >=16K on the uncapped cache and crashes on hybrid —
  upstream bug, §U); the zero-tolerance plain-greedy gate covers our
  loop at every tier it runs.
- v4 review amendments (2026-07-22, pre box-45 relaunch; 8-angle review
  of 711817f/12bf1da/b8a180e): (1) the long-tier arms run now carries
  --parity-check — the original v4 chain gated ONLY the 4K/8K smoke, so
  16K+ numbers would have been produced with no gate at their own tier;
  (2) make_cache fails LOUDLY if the offloading kwarg is rejected
  (silent fallback would relabel a non-offloaded run as offloaded);
  (3) the drafter's shared-KV view is moved to GPU OUTSIDE the timed
  window (the drafter forward otherwise pays the full-cache H2D copy
  inside draft_call_ms under offload — full arm charged GBs, sparse
  KBs: a PCIe artifact that would inflate the "sparse cheaper" gap);
  (4) SLIDING_CAP derived from config at runtime, not the hand-derived
  1025; (5) new draft_call_ms_warm column (drops each arm-run's first
  timed call: post-empty_cache allocator growth landed in call 0, and
  fixed arm order made "full" absorb it every time); (6) native
  cross-check now rejects runs where early EOS leaves <16 comparable
  tokens. CAVEATS THAT REMAIN BY DESIGN: at 16K+ the plain-greedy gate
  shares the offloaded cache + cap_sliding path with the loop (self-
  referential — a trunk-side semantics shift under the uncapped cache
  would be invisible to it); the offload-path anchor is the 4K/8K smoke,
  smoke logs have NO usable acceptance anchor (n=2 spread 1.17-1.48;
  see run_wp.sh gemma4g1 comment) — judge acceptance from n=6 tables.
  Do NOT compare 16K+ absolute draft-ms against the non-offloaded 4K/8K
  rows; within-tier arm gaps are the valid comparison.

### V2. 16K extension CERTIFIED (07-22 evening, box 46, runs dmeohmup
### smoke / xxk5nkqv arms @fa3bf57; CSVs rescue/2026-07-22-g1-16k/)

The ≥16K campaign burned three strategies before landing (full incident
log in the campaign memory + commits 7d99331/db29d06/408b167/99ad69d):
- **--offload RETIRED, upstream-broken**: DynamicCache(offloading=True)
  on transformers 5.15.0.dev0 is NONDETERMINISTIC past ~2K ctx (two
  identical plain-greedy chains diverge at token 0 at 4096, degenerate
  outputs; token-exact at 2000 — box-45 diag T1). Separately,
  return_shared_kv_states=True under offload corrupts the trunk forward
  (diag arms C/D). Never use offload for results until upstream fixes.
- **Chunked prefill RETIRED, uncertifiable**: source-level equivalence
  vs single-shot shows argmax same + max|dlogit| 0.672 but 13σ max
  outliers in the sliding shared-KV entry — cannot separate deep-layer
  bf16 amplification from the upstream continuation-mask bug.
- **Single-shot WORKS at 16K** with expandable_segments (v7 arms OOM'd
  on fragmentation: 674MB ask / 467MB free / 78.7GB held; the fix is
  allocator-policy only). Fit ladder: 16K peak 75.8GB (target-only),
  32K/65K OOM ⇒ **32K+ G1 remains BLOCKED on 1×H100** (all three
  routes exhausted); unblocking = 2×H100 + device_map=auto changes.

Gates (all green, run xxk5nkqv): zero-tolerance shape-aligned
plain-greedy parity 128/128 ×6 AT 16K (first long-tier gate ever run —
the review fix); native cross-check 32/32; smoke re-certification of
4K/8K on the identical code path (dmeohmup; draft-call 4.4/5.0ms
matches 92y2luja exactly). match_vs_full=1 on all 18 non-full rows.

| tier  | arm    | acc/round | acc_rate | draft ms | n |
|-------|--------|-----------|----------|----------|---|
| 16384 | full   | 1.244     | 0.249    | 6.28     | 6 |
| 16384 | sparse | 1.198     | 0.240    | 4.23     | 6 |
| 16384 | delta2 | 1.221     | 0.244    | 5.25     | 6 |
| 16384 | delta4 | 1.208     | 0.242    | 4.73     | 6 |

- **The cost gap grows exactly as predicted**: sparse drafter call is
  33% cheaper than full at 16K (4.23 vs 6.28ms), vs 16% at 8K and ~5%
  at 4K — full's read cost grows with cache length, sparse's is flat
  (4.2ms at every tier).
- **Acceptance near-parity at 18.75% visibility**: sparse 1.198 vs full
  1.244 acc/round = 96.3% of full. HONEST CAVEAT: unlike 8K (where
  sparse was a nose ahead), at 16K the paired per-prompt diffs are
  full−sparse = +0.018/+0.075/0/+0.094/0/+0.088 — 4/6 positive, 2 exact
  ties, 0 negative: a small but CONSISTENT full edge appears at 16K.
  Do not say "indistinguishable" for 16K; say ~96% of full acceptance
  at ~1/5 visibility and 2/3 the read cost. delta2 recovers about half
  the gap (1.221, 98.2%); delta4 sits between (1.208, 97.1%).
- Acceptance is NOT monotone in ctx across tiers here (16K full 1.244 >
  8K 1.18) — different PG19 docs per tier; only within-tier arm
  comparisons are controlled.
- Smoke-vs-table acc note: the smoke rows (max_new 64, n=2) read lower
  at 4K (1.48/1.20) than the certified table mean (1.97, max_new 128,
  n=6) — parameter mismatch, not a regression; smoke certifies gates
  and timing, the n=6 tables are the quotable acceptance numbers.


---

# Part II — pre-pivot archive

Compiled 2026-07-15. Raw numbers only — no derived analysis except where a
quantity is explicitly defined so it can be recomputed.

**Shared eval grid:** all perplexity evals use the SAME 32 deterministic
held-out PG19-test chunks at 32,768 tokens (selection is deterministic, so
values pair across every run in this file and in
`update_data_2026-07-12.md`/`update_data_2026-07-14.md`).
loss = mean token cross-entropy in nats; ppl = exp(loss).
"Paired diff" = per-chunk loss difference between two arms on identical
chunks: mean ± sem (n=32), plus the count of chunks favoring the
first-named arm.

---

## A. 32K retraining

3 arms × 500 steps @ seq_len 32,768 — token count equal to the 8K pilot
(2000 × 8192). Identical LoRA (r16 α32 q/k/v/o), lr 1e-4→1e-5 cosine,
shuffled PG19, same seed. Adapters: `wp2_adapter_{delta,dense,detach}_32k`.

### A1. Pipeline eval (run `1e777qkb`), 8K-trained arms on same chunks for reference

| arm | 32K-trained ppl | 8K-trained ppl |
|---|---|---|
| base (no training) | 12.2219 | 12.2219 |
| delta-trained | 10.4067 | 10.3732 |
| detach-trained | 10.5474 | 10.4691 |
| dense-trained | 10.6680 | 10.5851 |

Paired diffs, 32K-trained arms, pipeline eval:

- delta − dense: **−0.0248 ± 0.0014, delta better on 32/32**
  (8K-trained equivalent: −0.0202 ± 0.0011, 32/32)
- delta − detach: −0.0134 ± 0.0006, 32/32
- detach − dense: −0.0114 ± 0.0010, 32/32
- base − delta: +0.1608; base − detach: +0.1474; base − dense: +0.1360

### A2. Dense eval (run `s9rxrd1m`)

| arm | 32K-trained ppl | 8K-trained ppl |
|---|---|---|
| base | 11.2652 | 11.2652 |
| dense-trained | 9.7907 | 9.7333 |
| detach-trained | 9.9365 | 9.7738 |
| delta-trained | 9.9616 | 9.7946 |

- delta − dense under dense eval: **+0.0173 ± 0.0010, delta better on 0/32**
  (8K-trained: +0.0063)

### A3. Pipeline tax (defined: ln(pipeline ppl) − ln(dense ppl), same model)

| arm | tax |
|---|---|
| base | 0.0815 |
| dense-trained 8K / 32K | 0.0839 / 0.0858 |
| detach-trained 8K / 32K | 0.0687 / 0.0597 |
| delta-trained 8K / 32K | 0.0574 / 0.0437 |

### A4. Ops

Wall-clock: delta 54 min, dense 68 min, detach ~53 min (1×H100).
Final delta training loss 2.2574.

---

## B. Anchor-gradient decomposition (Jeff's 1/γ question)

32K delta training run `etoxgd7p`. Probe: layer-0 query gradients through
the differentiable delta forward (γ=64) on a held-out 32K batch, every 50
steps → 10 probes.

- anchor_grad_ratio, full graph: **mean 3.405, range 3.373–3.427**
- anchor_grad_ratio, correction term detached: **0.955**
  (constant to 3 decimals across all 10 probes)
- 8K reference (the value Jeff has seen): 1.466, range 1.457–1.476
  (20 probes)

---

## C. Distillation (three variants)

All trained @8K, 2000 steps, same data/seed as the CE pilot. Student = base
+ LoRA under the differentiable delta pipeline. Teachers:

- `distill`: frozen base model, adapters disabled, dense attention; pure KL
- `distill_mix`: same teacher; loss = KL + 1.0 × CE
- `distill_dft`: separate frozen model with the dense-CE (pilot_dense) LoRA
  merged in, dense attention; pure KL. NOTE: teacher fetched as
  `wp2_adapter_dense:latest` (UNPINNED at run time — pin before any rerun)

### C1. Full 7-arm 2×2 (pipeline run `bt676j1m`, dense run `bc3z02ye`)

| arm | pipeline ppl | dense ppl | tax |
|---|---|---|---|
| base | 12.2219 | 11.2652 | 0.0815 |
| dense-CE | 10.5851 | 9.7333 | 0.0839 |
| detach-CE | 10.4691 | 9.7738 | 0.0687 |
| delta-CE | 10.3732 | 9.7946 | 0.0574 |
| distill (pure KL, base teacher) | 11.7148 | 11.1852 | 0.0463 |
| distill_mix (KL-base + CE, α=1) | 10.6547 | 10.1079 | 0.0527 |
| distill_dft (pure KL, dense-ft teacher) | 10.3732 | 9.8418 | 0.0526 |

### C2. Paired diffs (pipeline eval unless noted)

- distill − base: −0.0424 ± 0.0020, 32/32
  (under dense eval: −0.0071 ± 0.0006, 32/32)
- distill_dft − delta-CE: **+0.00001 ± 0.00021, dft better on 13/32**
  (per-chunk losses bitwise-equal on 0/32 chunks → genuinely different
  models; the mean-level tie is real, not an eval artifact)
- distill_dft − dense-CE: −0.0202 ± 0.0012, 32/32
- distill_dft − distill: −0.1216 ± 0.0032, 32/32
- distill_mix − delta-CE: +0.0268 ± 0.0010, mix better on **0/32**
- distill_mix − dense-CE: +0.0066 ± 0.0014, mix better on 4/32

### C3. Ops

Training wall-clock: distill 57 min, mix 59 min, dft 56 min.

---

## D. InfiniteBench En.MC

longbook_choice_eng, n=229 (full subtask), scored by argmax over
logprobs("A","B","C","D") at the last prompt position. Book contexts
middle-truncated (head+tail) to 31,500 tokens. Run `44cqxfpv`.

| arm | accuracy |
|---|---|
| base_dense | 0.6114 |
| ce_delta (8K delta-trained) | 0.5677 |
| dense_delta (8K dense control) | 0.5590 |
| base_delta | 0.5546 |

Chance = 0.25. Binomial sem at these accuracies, n=229 ≈ 0.033.
No per-sample logs in this run (capability added to the code afterward) →
aggregate comparison only, no paired test possible.

---

## E. γ_dec sweep, LongBench QA (Jeff's "between 2 and 16" question)

Run `rwiomfcx`, n=50/task. All decode arms: base model, delta γ=64 prefill,
only decode differs. New arms, per task (F1):

| task | γ_dec=4 | γ_dec=8 |
|---|---|---|
| hotpotqa | 0.4459 | 0.4200 |
| 2wikimqa | 0.4105 | 0.4105 |
| musique | 0.3009 | 0.2922 |
| multifieldqa_en | 0.5744 | 0.5565 |
| **mean** | **0.4329** | **0.4198** |

Existing reference means on the same setup (07-13 runs, already sent to
Jeff): dense decode 0.454, γ_dec=2 0.4345, γ_dec=16 0.4148, pure sparse
0.417, base_dense 0.469. (Per-task values for these arms are in wandb.)

Full decode curve, means:

| decode | mean F1 |
|---|---|
| dense | 0.4540 |
| γ_dec=2 | 0.4345 |
| γ_dec=4 | 0.4329 |
| γ_dec=8 | 0.4198 |
| γ_dec=16 | 0.4148 |
| pure sparse | 0.4166 |

RULER 32K reference for the same decode ladder (07-12 decsweep runs
qowey1wk/p51p93rz/l6l4cokx/0l2xm988; n=150, fixed 3-task subset): dense
76.7, γ_dec=1 78.0, γ_dec=2 28.67, γ_dec=4 26.67, γ_dec=8 26.67, γ_dec=16
24.7, γ_dec=64 25.33, pure sparse 20.0. (CE-ft arm at γ_dec=16: 20.0.)
Per-task: niah_single_1 = 8.0 and niah_multikey_2 = 24.0 at EVERY γ — all
curve variation is qa_1 (54→48→48→44); needle retrieval is flat-dead at
any reuse ≥ 2, not gradually degrading.

---

## F. GovReport free-running decode sweep (run `23o7tkqf`, n=50, 512 new tokens, ROUGE-L f)

| arm | ROUGE-L | fraction of dense decode |
|---|---|---|
| base_delta (delta prefill + dense decode) | 0.3476 | 1.000 |
| base_dense (fully dense) | 0.3470 | — |
| delta γ_dec=2 | 0.3296 | 0.948 |
| delta γ_dec=4 | 0.3218 | 0.926 |
| delta γ_dec=8 | 0.3164 | 0.910 |
| delta γ_dec=16 | 0.3128 | 0.900 |
| pure sparse | 0.3068 | 0.883 |

Paired per-sample diffs (n=50, from `longbench_samples.csv`, archive v12):

- base_delta − base_dense: +0.0006 ± 0.0044, 26/50 — delta PREFILL is free
  on summarization
- delta γ2 − dense decode: −0.0180 ± 0.0055, dec2 better on 14/50
- delta γ4 − dense decode: −0.0258 ± 0.0056, 11/50
- **delta γ4 − pure sparse: +0.0151 ± 0.0055, dec4 better on 33/50** — the
  correction adds real value over sink+window on long free-form generation
- delta γ16 − pure sparse: +0.0061 ± 0.0046, 28/50 (fades at long reuse)

Figure: `figures/jeff_decode_by_task.png` (QA/GovReport/RULER decode curves
normalized to each task's dense-decode score).

## G. dft+CE (distill_dftmix) — the ceiling breaks (pipeline `oukke8tg`, dense `2igy21mx`)

KL(dense-ft teacher) + 1.0·CE, same pilot dials. Same 32 chunks.

| arm | pipeline ppl | dense ppl | tax |
|---|---|---|---|
| delta-CE | 10.3732 | 9.7946 | 0.0574 |
| distill_dft (pure KL) | 10.3732 | 9.8418 | 0.0526 |
| **distill_dftmix (KL+CE)** | **10.3565** | 9.7993 | 0.0553 |

Paired (pipeline): dftmix − delta-CE = **−0.00161 ± 0.00016, better on
32/32 chunks (~10 sem)**; dftmix − dft = −0.00161 ± 0.00019, 30/32.

Readings: (1) first objective to beat CE-through-the-pipeline — the 10.37
"ceiling" was an objective limitation, not pure capacity, though the
breach is small (~0.16% ppl); (2) the KL×CE interaction FLIPS SIGN with
teacher adaptation: base-teacher KL+CE is worse than plain CE on 32/32
(+0.027), dense-ft-teacher KL+CE is better on 32/32 (−0.0016); (3) dftmix
= delta-CE-level dense adaptation + most of dft's tax reduction.

## H. Benchmark transfer: 32K-trained + distill_dft arms (runs `xm2j1gaa` v1 / `mtu4s1s9` enmc)

8 arms, identical prompts, per-sample logged (paired tests possible from
the archived `longbench_samples.csv`, box_final_state artifact of the
bench32k box).

| arm | LongBench QA mean F1 (4 tasks × n=50) | En.MC acc (n=229) |
|---|---|---|
| base_dense | 0.4693 | 0.6114 |
| base_delta | 0.4539 | 0.5546 |
| ce_delta (8K) | 0.4512 | 0.5677 |
| dense_delta (8K) | 0.4626 | 0.5590 |
| ce32k_delta | 0.4492 | 0.5415 |
| dense32k_delta | 0.4413 | 0.5677 |
| detach32k_delta | 0.4705 | 0.5328 |
| distill_dft_delta | 0.4601 | 0.5546 |

Notes for interpretation: base/8K arms replicate the 07-13/14 aggregates to
±0.003 (instrument stability); no arm shows a CONSISTENT edge across both
instruments (detach32k is best on QA and worst on En.MC); binomial sem on
En.MC ≈ 0.033; QA mean-F1 sem ≈ 0.01–0.015. Nothing collapsed relative to
base_delta.

Paired per-sample tests (from the rescued `_samples.csv`; identical
prompts, diff = first − second arm per item):

| comparison | QA (n=200) | En.MC (n=229) |
|---|---|---|
| ce32k − base_delta | −0.005 ± 0.027 | −0.013 ± 0.029 |
| ce32k − dense32k | +0.008 ± 0.024 | −0.026 ± 0.025 |
| distill_dft − ce_delta | +0.009 ± 0.015 | −0.013 ± 0.016 |
| detach32k − base_delta | +0.017 ± 0.025 | −0.022 ± 0.027 |

Every comparison < 1.1 sem from zero, and most items are exact TIES
(65–95% of samples produce identical scores across arms) — the transfer
null is paired-test-grade, not an aggregate artifact.

## I. Speculative decoding harness (specdec2) — method + certification, results in flight

Implementation (eval/specdec_eval.py): greedy draft-and-verify on ONE
shared KV cache. Draft = sparse or delta decode proposes K tokens (delta
draft resets per block: one dense-row anchor + K−1 delta reuses — γ_dec
maps to block size); verify = ONE dense rectangle forward over the block;
sequential acceptance until first mismatch (dense token substituted);
bonus token on full acceptance; cache cropped to dense-grade length before
each verify and past the first rejection after.

Certification: emitted tokens are dense-greedy by construction. Bitwise
parity with one-token-at-a-time dense decode is unattainable (verify
writes KV with q_len=K kernels; bf16 reduction order differs; a near-tie
argmax flips LATE — measured: rounds 1–15 exact, first flip at token 75;
a 128-token blocked rescore matched sequential 128/128, exonerating the
bookkeeping). Smoke gate `--min-parity-prefix 24` PASSED on box 15:
systematic harness bugs diverge at tokens 0–10; ours diverges only at
kernel-numerics depth.

### I1. Base-model acceptance grid, GovReport (run `srokwtvo`, n=20, 512 new tokens)

| draft | block | acceptance | accepted/verify | full-block rate |
|---|---|---|---|---|
| sparse | 2 | 0.7670 | 1.53 | 0.704 |
| sparse | 4 | 0.6317 | 2.53 | 0.501 |
| sparse | 8 | 0.4335 | 3.47 | 0.270 |
| delta | 2 | **0.9429** | 1.89 | 0.890 |
| delta | 4 | 0.8140 | 3.26 | 0.649 |
| delta | 8 | 0.5909 | 4.73 | 0.334 |

- **Wall-clock speedup claims RETRACTED pending rerun** (integrity audit
  2026-07-15 evening): run srokwtvo started 59s before the fair-timing fix
  landed, so its spec tok/s excluded prefill while dense tok/s included it,
  and neither side CUDA-synchronized its timers — the previously-quoted
  1.24–1.37× numbers are inflated by construction and are NOT to be quoted.
  Acceptance/accepted-per-verify/full-block above are timing-independent
  and stand. A clean timing rerun (synced timers, fair populations) is
  queued.
- parity_prefix_min per config: 75–118 (all divergences at kernel-numerics
  depth; output is dense-greedy by construction).
- **Acceptance decomposition (scoped review, 07-16):** the delta draft's
  FIRST proposal per block comes from its per-block dense anchor step
  (state resets each block → the pending-token forward is a true dense
  row), so position 1 is accepted ~always by construction. Genuine
  drafted-position acceptance (positions 2..K, assuming pos-1 ≈ 1):
  base K=2 0.886, K=4 0.752, K=8 0.532; ce32k K=4 0.872; dft K=4 0.851 —
  still above sparse everywhere, but THESE are the values comparable to
  small-draft literature (EAGLE-style drafts get no free dense position),
  and the per-block dense-row cost must be charged to the delta draft.
- Protocol caveats for external comparison: acceptance includes proposals
  in a final block past an emitted eos/max_new cut; the QA spot-check used
  a 64-token budget for all tasks (official LongBench budgets are 32 for
  3 of the 4); the parity gate ran on the smoke only, not the grid stages.
  All fixed in the harness for the planned clean rerun.
- Delta draft beats the sparse draft by +15.7–18.2 acceptance points at
  every block size — the delta correction's first large, unambiguous win, in
  exactly Jeff's speculative frame. Delta K=2 ≈ 2.9 tokens per dense call.
- Wall-clock caveat for the writeup: our draft kernels are unoptimized
  Torch paths; acceptance is the implementation-independent quantity, the
  1.37× is a floor, not a ceiling.

### I2. Trained drafts + QA spot-check (runs `onj1sptp`, `ok0bwxjm`)

GovReport, block 4 (base reference: delta 0.8140 / sparse 0.6317):

| weights | draft | acceptance | accepted/verify | full-block | parity_min |
|---|---|---|---|---|---|
| ce32k (32K delta-trained) | delta | **0.9040** | 3.62 | 0.822 | 174 |
| ce32k | sparse | 0.8057 | 3.22 | 0.726 | — |
| dft (dense-ft-teacher distilled) | delta | 0.8886 | 3.55 | 0.785 | 45 |
| dft | sparse | 0.7599 | 3.04 | 0.671 | 45 |

(ce32k/sparse metrics recovered from run onj1sptp's logged history:
acceptance 0.8057, accepted/verify 3.223, full-block 0.7259. Only its
parity_min existed solely in the unarchived box CSV.)

LongBench QA, block 4, base weights (n=40): delta draft 0.8207 vs sparse
0.7596; parity 'full' (byte-identical to dense on all checked prompts).

Readings: (1) training through the pipeline IMPROVES the draft — ce32k adds
+9.0 acceptance points over the base delta draft (~4.6 tokens per dense
call at 90% acceptance); (2) training helps even the PURE SPARSE draft
(ce32k/sparse 0.806 and dft/sparse 0.760 vs base/sparse 0.632 — +17.4 and
+12.8 points) — Jeff's "SWA tokens learn to deal with the delta
information" conjecture, in data; (3) the
benchmark-transfer null and this result are two sides of one coin:
pipeline training buys dense-AGREEMENT, not generic quality — and
agreement is exactly what speculative drafting pays for. WP-2 and WP-3
compose.

### I3. Definitive v2 rerun (specdec3, 2×H100 box, 07-16) — fixed harness:
synced timers, official QA budgets, eos-boundary acceptance, parity gate on
every stage, per-position logging, dftmix draft

**QA grid (run `ufqln1td`, n=40, official budgets 32/32/32/64): COMPLETE,
parity `full` on every checked prompt in all 24 cells.** Coverage caveat
(07-16 review): the 3 checked prompts per cell all come from the FIRST QA
task and hit natural eos after 3–7 generated tokens, so "byte-identical"
covers ~15 tokens of one task per cell — real evidence, but NOT end-to-end
exactness certification across tasks/lengths; the GovReport parity prefixes
(45–436 tokens) are the strong certification. Full table in
`rescue/2026-07-16/sd3_qa.csv` (+ per-sample `sd3_qa_samples.csv`).
Headline / genuine acceptance (genuine = positions 2..K only):

| weights | draft | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| base | sparse | 0.810 / 0.752 | 0.643 / 0.586 | 0.401 / 0.350 |
| base | delta | 0.912 / 0.824 | 0.727 / 0.636 | 0.495 / 0.423 |
| ce32k | sparse | 0.798 / 0.774 | 0.646 / 0.604 | 0.375 / 0.335 |
| ce32k | delta | 0.953 / 0.905 | 0.798 / 0.730 | 0.537 / 0.470 |
| dft | sparse | 0.866 / 0.851 | 0.737 / 0.711 | 0.421 / 0.370 |
| dft | delta | 0.953 / 0.906 | 0.829 / 0.781 | 0.488 / 0.415 |
| dftmix | sparse | 0.856 / 0.849 | 0.730 / 0.711 | 0.478 / 0.441 |
| dftmix | delta | 0.944 / 0.887 | 0.838 / 0.784 | 0.550 / 0.486 |

- The delta draft's measured position-1 acceptance is **1.0000 in 11/12
  delta cells** (0.9714 in dft/b4) — the structural dense-anchor accept,
  now measured rather than argued. Sparse pos-1 is 0.66–0.88.
- Per-position curves are monotone decreasing in every cell (prefix
  acceptance is monotone by construction; curves in the CSV `pos_acc` col).
- QA tok/s is prefill-dominated at 32-token budgets — not a speedup claim.
  (base-weights `tok_per_s_dense` 3.16 vs ~6.7 for adapter configs is the
  cold-process artifact: refs are timed immediately after the FIRST model
  load of the process — the same bias that inflated the GovReport speedups,
  at full strength here. QA timing should not be quoted at all.)

**GovReport grid (runs `re9piix8`, `4oqfl598`, `32s636u0`, `jaihgxn9`;
n=20, 512 new tokens): COMPLETE.** Mid-run incident: the original parity
gate hard-failed cell ce32k/sparse/b8 at a token-17 divergence. Margin
probe on the box: top-2 logit margins at the three checked-prompt
divergence points were **0.125 / 0.000 / 0.250** (0–2 bf16 ulps; in 2/3
cases the probe sided with the SPEC output against the sequential
reference). PROVENANCE + METHOD CAVEATS (07-16 review): the original
`~/margin_probe.log` was lost with the box — the transcript copy is at
`rescue/2026-07-16/margin_probe.log` — and the probe ran under the
window/delta APPROXIMATE path, not dense attention (the "fresh dense
forward" wording was wrong), so the tie interpretation is probable
(corroborated by the rerun's 174-token certified prefix on the same
config) but not proven. Gate v4 (`ae6a30c`): early divergences get a
one-forward margin check; margin ≤ 0.5 → recorded tie, run continues;
larger → hard fail. The rerun cell carries `parity 174+1tie(17@0.12)`.
Gate v5 (07-16, post-review): probe fixed to true verify-path semantics
(pipeline prefill + dense rectangle), ties now require a corroborating
certified prompt, -1 fails unconditionally.

Headline / genuine acceptance (parity mins 45–436 across all cells):

| weights | draft | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| base | sparse | 0.766 / 0.703 | 0.630 / 0.573 | 0.429 / 0.382 |
| base | delta | 0.943 / 0.889 | 0.813 / 0.753 | 0.587 / 0.529 |
| ce32k | sparse | 0.884 / 0.850 | 0.804 / 0.773 | 0.682 / 0.655 |
| ce32k | delta | **0.968 / 0.939** | **0.903 / 0.873** | **0.783 / 0.752** |
| dft | sparse | — | 0.758 / 0.720 | — |
| dft | delta | — | 0.888 / 0.851 | — |
| dftmix | sparse | — | 0.762 / 0.729 | — |
| dftmix | delta | — | 0.886 / 0.850 | — |

- Delta pos-1 on GovReport: 0.9931–0.9974 everywhere (structural anchor
  accept). ce32k/delta/K=8: accepted-per-counted-verify 6.26 → **τ ≈ 7.26**
  (the CSV's 6.18 used a mixed denominator — cut final blocks counted in
  verifies but not in accepted; fixed in the harness 07-16. On QA the same
  bug understates apv ~2×; ignore the QA apv column from run ufqln1td).
- **Timing — CORRECTED twice; FINAL numbers below (07-16 review, then
  sdtiming rerun).** History: the as-run 1.20–1.22× divided warm spec
  numerators by COLD dense baselines (refs timed right after model load;
  both quoted ce32k cells came from resume runs where ce32k was first —
  dense 24.81/24.83 cold vs 26.23 warm for the identical config), AND the
  dense side paid generate() per-step machinery the lean spec loop doesn't
  (~8–12%: lean warm dense measures 27.5–29.6 tok/s vs generate()'s
  24.8–26.2). Do not quote anything from runs
  re9piix8/4oqfl598/32s636u0/jaihgxn9's tok/s columns.

**DEFINITIVE timing (sdtiming run, box 29: runs `v0hw2c6x` delta grid,
`chh7xuen` sparse contrast; --warm-baseline: untimed warmup, then a LEAN
sequential dense loop — same machinery as the spec loop, synced — timed
over ALL 20 prompts per config, spec timed on all 20; GovReport n=20,
512 tok, 32K, batch 1, single H100 sxm5, GPU1 idle). Acceptance replicated
the specdec3 grid EXACTLY (deterministic greedy). Parity mins 53–436 under
gate v5:**

| draft | K | tok/s spec | tok/s dense (lean, warm, same config) | speedup |
|---|---|---|---|---|
| base delta | 2 | 32.17 | 27.45 | **1.17×** |
| base delta | 4 | 30.28 | 27.45 | 1.10× |
| base delta | 8 | 23.56 | 27.45 | 0.86× |
| ce32k delta | 2 | 32.63 | 28.96 | 1.13× |
| ce32k delta | 4 | 32.86 | 28.96 | **1.13×** |
| ce32k delta | 8 | 30.16 | 28.96 | 1.04× |
| dftmix delta | 2 | 32.52 | 29.31 | 1.11× |
| dftmix delta | 4 | 32.47 | 29.31 | 1.11× |
| dftmix delta | 8 | 28.26 | 29.31 | 0.96× |
| base sparse | 4 | 25.86 | 27.68 | 0.93× |
| ce32k sparse | 4 | 31.06 | 29.15 | 1.07× |
| dftmix sparse | 4 | 29.81 | 29.55 | 1.01× |

  - Headline for external use: **~1.1–1.15× wall-clock at 32K/batch-1**
    (same-size self-speculative draft; the ceiling is the attention share
    of decode cost). τ (accepted-per-counted-verify + 1): ce32k K=8 →
    **7.26**, K=4 → 4.61.
  - Residual caveat: dense lean baselines still drift upward with config
    position within a process (27.45 → 28.96 → 29.31 in run order, both
    runs) — allocator/clock maturation. Within-config ratios (what the
    table shows) are warm-vs-warm and fair; CROSS-config speedup
    comparisons carry ±0.03–0.05 of this drift (e.g. base-delta's 1.17×
    vs ce32k's 1.13× is partly its lower first-position baseline).
  - Trained drafts don't raise the peak speedup (acceptance gains ≈ offset
    by nothing at K≤4) but they RESCUE large K: K=8 goes 0.86× (base) →
    1.04× (ce32k), and sparse drafting goes 0.93× → 1.07×. Raw CSVs:
    `rescue/2026-07-16/sdtiming/` + wandb box_final_state:v19.

**RULER negative control (run `s0ssjsnj`, niah_single_1 @32K, n=10, base
weights): parity `full` (byte-identical), and acceptance collapses as
predicted.** Sparse genuine acceptance 0.270 (K=4) / 0.148 (K=8) — near
floor. Delta keeps its structural pos-1 accept (1.0000) but decays fast
after: per-position K=4 curve 1.00;0.76;0.46;0.35 (genuine 0.523 vs 0.75
on GovReport). Jeff's "impossible by construction" prediction holds at the
suite level. Mechanism NOTE (07-16 review): "the needle span is exactly
where it dies" is an INFERENCE from the positional curve — nothing logged
localizes rejections to needle-content vs answer-template tokens; a
token-level rejection log would be needed to state it as measured.
Also: sd3_gov_samples.csv contains 20 rows for ce32k/sparse/b8 from the
gate-aborted re9piix8 attempt alongside the 4oqfl598 rerun's 20 — filter
by run_id for any per-sample analysis.

## K. Gradient-scale arms — Jeff's 1/γ idea tested interventionally

Two 32K delta arms with the correction branch's BACKWARD gradient scaled
(forward math unchanged), same dials/chunks as train32k. Arm-faithful
probe confirmed the intervention: achieved anchor_grad_ratio ≈ 1.10 under
1/√γ (vs 3.41 unscaled; probes 1.102–1.104). Eval run manually on the held box during the
wandb artifact outage (metrics sync unaffected); adapters rescued to
`~/Development/delta-attention/rescue/2026-07-15/`.

| correction-grad scale | pipeline ppl | dense ppl | tax |
|---|---|---|---|
| 1.0 (delta_32k ref) | 10.407 | 9.962 | 0.0437 |
| 1/√γ (0.125) | 10.482 | 9.906 | 0.0565 |
| 1/γ (0.0156) | 10.538 | 9.932 | 0.0592 |

Paired (PIPELINE eval, run w2mqmov9): gsqrt − gsinv = −0.0053 ± 0.0002
(gsqrt better); base − gsqrt = +0.1536 ± 0.0040. (Dense-eval equivalents:
−0.0026 ± 0.0001 and +0.1286 ± 0.0032.) Cross-run pairing to delta_32k
valid via the deterministic chunks. Reading: damping the correction gradient hurts
MONOTONICALLY in damping strength — the anchor-gradient concentration is
beneficial delta-specific signal, not a pathology. Interventional
confirmation of the observational dose-response; no 1/γ backward needed.

## L. MMLU capability retention (run `bt0m1r4k`, n=1000, letter-logprob, all arms force-dense)

| arm | acc | | arm | acc |
|---|---|---|---|---|
| base | 0.6380 | | detach32k | 0.6490 |
| ce_delta (8K) | 0.6460 | | distill | 0.6400 |
| dense_delta (8K) | 0.6420 | | distill_mix | 0.6400 |
| ce32k | 0.6400 | | distill_dft | 0.6470 |
| dense32k | 0.6400 | | | |

Every trained arm ≥ base; spread 0.638–0.649 ≈ noise (sem ~0.015). NO
capability damage from any training variant.

## M. Directional-bias probe (run `qczzak90`; anchor-vs-window dL/dW vs LoRA
update ΔW=B·A, per q/k/v/o × layers 0/8/16/24/31; CSV rescued locally)

| arm | cos(ΔW,g_anchor) | cos(ΔW,g_window) | cos(gA,gW) | |gA|/|gW| |
|---|---|---|---|---|
| delta (8K) | −0.001 | −0.001 | 0.039 | 4.61 |
| delta_32k | +0.001 | −0.003 | 0.040 | 4.52 |
| detach | −0.001 | −0.001 | 0.102 | 5.07 |
| dense | −0.001 | −0.001 | 0.030 | 4.59 |

Anchor gradients are ~4.6× stronger and nearly orthogonal to window
gradients, yet the learned update aligns with NEITHER population and shows
no anchor tilt — Jeff's directional-dominance concern is not realized in
the final update. Caveat: endpoint-gradient probe; cannot rule out
path-dependent dominance early in training.

## J. Status (final update ~6:30pm 07-15)

**Running:** the two seed-replication chains ONLY — seedsdistill (box 26:
delta-CE vs distill_dft × seeds 1,2 @8K + paired 2×2) and seeds32k (box 27:
delta vs dense × seeds 1,2 @32K + paired 2×2), ~5h each, self-terminating.
Everything else in this file is COMPLETE.

**Ops notes:** keys ROTATED (the afternoon "wandb outage" was the rotation:
old key kept metadata alive while GCS refused its signed URLs and uploads
silently no-committed; two gradscale adapters were rescued via scp and
restored). All 12 adapters now mirrored to the private HF repo
`singhh5050/delta-attention-adapters`; fetch_adapters falls back to it.

**Known not-yet-run:** benchmarks + spec-decode acceptance for the dftmix
adapter; RULER negative-control spec run; KL-weight sweep and layer-output
distillation (deferred per plan stopping rules).

## N. Seed replication (overnight 07-15→16; both boxes self-terminated)

**seeds32k** (run `s9qiqibn`; delta/dense retrained @32K with data seeds 1,2):

| arm | seed 0 (orig) | seed 1 | seed 2 |
|---|---|---|---|
| delta_32k | 10.4067 | 10.4118 | 10.4003 |
| dense_32k | 10.6680 | 10.6821 | 10.6719 |

Cross-seed paired delta_s1 − dense_s2 = −0.0247 (replicates −0.0248). The
32K delta-vs-dense gap is fully seed-robust; seed-to-seed movement of
individual arms is ±0.006 ppl vs a gap of ~0.26.

**seedsdistill** (run `9bzq16ox`; delta-CE and distill_dft retrained @8K,
seeds 1,2):

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| delta-CE | 10.3732 | 10.3712 | 10.3565 |
| distill_dft | 10.3732 | 10.3703 | 10.3573 |

- THE TIE REPLICATES PER-SEED to ±0.0001: paired delta − dft = +0.00009
  (seed 1), −0.00008 (seed 2), +0.00001 (seed 0).
- Absolute LEVELS move with seed (delta_s1 − delta_s2 = +0.00142 loss —
  seed variance ~±0.0014), but same-seed DIFFERENCES are stable to
  ±0.0001 — arms co-move with data order. All of this file's paired
  claims are difference-based, i.e. the seed-stable quantity.
- Consequence for §G: dftmix's −0.00161 vs delta-CE (same seed, same data)
  is a same-seed difference — the seed-stable kind — and sits ~10× above
  the ±0.0001 seed noise of such differences. The ceiling-break claim
  survives replication analysis.

## OLD §J snapshot (superseded, kept for the timeline)

**Lambda (2-box hard cap, user rule):**

| box | mode | stage | produces | ETA |
|---|---|---|---|---|
| harsh experiment 15 | specdec2 | base acceptance grid (sparse+delta drafts × blocks 2/4/8, GovReport n=20) | acceptance rate, accepted-per-verify, full-block rate, fair tok/s, parity | grid ~4:30pm, then trained drafts (ce32k, dft @ block 4) + QA spot-check → ~6pm |
| harsh experiment 21 | gradscale | setup | 1/√γ and 1/γ correction-gradient arms @32K (Jeff's idea, interventional) + paired 5-arm 2×2 | ~5:30pm |

**Queued (sniper rotates through freed slots, never exceeding 2):**
1. mmlu — 9 arms × 1000 MMLU (capability retention; force-dense) + the
   directional-gradient probe (anchor-vs-window dL/dW vs LoRA update
   direction, per q/k/v/o, layers 0/8/16/24/31) — ~2h once slotted
2. seedsdistill — delta-CE vs distill_dft × seeds 1,2 @8K + paired 2×2
   (variance behind the 10.37 tie/break) — ~5h
3. seeds32k — delta vs dense × seeds 1,2 @32K + paired 2×2 (variance behind
   the −0.025 gap) — ~5h

**Brev:** zero instances (all stragglers deleted); no H100 capacity on any
provider at last check; `brev login --token` sessions last exactly 15 min
(placeholder refresh token) — durable use needs the interactive browser
login.

**Not yet run (known gaps):** En.MC/LongBench on the dftmix adapter;
spec-decode acceptance for dftmix; RULER negative-control spec run;
key rotation (user-side).

## O. Paper-core triad + training-efficiency track — MOVED

The 07-17 triad (T1 training-efficiency benchmark, T2 downstream
retention, T3 arXiv replication), the 07-20 review corrections, and the
definitive O4 clean trainbench rerun now live in
`stats_training_mtp.md` (same section numbering), along with all future
training-efficiency and MTP spec-dec experiments. This file remains the
record of the pre-pivot work (through spec-decode v2).
