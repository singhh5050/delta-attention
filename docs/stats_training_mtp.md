# Training-efficiency & MTP spec-dec track — raw stats

Started 2026-07-17 with the paper pivot (Jeff's plan: triad of same-ppl /
same-downstream / cheaper-training, plus delta-in-MTP as the spec-dec
path forward). Same conventions as `stats_2026-07-15.md`: raw numbers
only, every value with its wandb run ID; paired = per-chunk/per-sample
diffs on identical inputs, mean ± sem.

Earlier project record (spec decode, distillation, 32K training, seeds,
gradient probes): `stats_2026-07-15.md`. Quick reference:
`PROJECT_CHEATSHEET.md`.

---

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

