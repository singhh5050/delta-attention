# Experiments completed since the last Jeff update

Scope: everything AFTER the 07-22 ~1:03 AM messages (anchor-branch
diagnosis + flexrow fix + 131K–1M ladder + Qwen3-14B replication — all
already sent), plus two MTP results that were run on 07-21 but never
sent. Jeff's stated next step: "wait for the MTP experiment to finish"
then pitch to his manager. Sections 1–3 ARE that MTP story.

Every number below has run-ID provenance in docs/stats_training_mtp.md
(section cited per item); raw CSVs triple-archived (wandb run +
box_final_state artifact + local rescue/).

---

## 1. Gemma 4 native MTP drafter × delta reads (THE awaited experiment)
Stats: §U (G0), §V + §V2 (G1). Plan/wiring: docs/gemma4_mtp_plan.md.

**Why Gemma 4**: every Gemma 4 model ships a jointly-trained multi-token
drafter invoked via vanilla HF assisted decoding (~3× claimed speedup).
Verified from the modeling source + checkpoint weights: the drafter has
NO K/V projections at all — 4 layers of pure cross-attention (q_proj
only) over the TRUNK's cached KV, no drafter-side cache. Delta-ifying it
= subsampling the trunk-KV view it is handed. No model surgery.

**G0 — native baseline certified** (runs qobgwn2z/vx6ldm79):
- Found an upstream transformers bug on the way: assisted decoding
  crashes for prompts past the trunk sliding window (hybrid cache
  window-caps sliding KV vs full-length verify mask; sdpa AND eager).
  Worked around with an explicit DynamicCache, parity-gated.
- Native drafter decode-only speedup (two-length differencing, both
  modes on identical cache semantics): median 1.43× @4K, 1.36× @8K on
  the naive single-prompt HF path (occasional fully-accepted prompts
  ~4×; Google's ~3× claim presumably reflects a served stack).

**G1 — delta-reading the drafter (the result)** (runs 92y2luja @711817f
and dmeohmup/xxk5nkqv @fa3bf57):
- Our own draft-verify loop, transcribed from the HF candidate
  generator; the drafter is never modified — arms differ ONLY in the
  shared-KV view: full / sparse (sink1024+window2048) / deltaN (full
  read every Nth drafter call).
- Gates all green, including a ZERO-tolerance parity gate vs plain
  trunk greedy computed with identical kernel shapes (64/64 ×4 at
  4K/8K; 128/128 ×6 at 16K) + an independent short-context native
  cross-check (32/32). Acceptance is trunk-verified = exact.

| tier | visible to sparse | full acc/round | sparse | delta2 | delta4 | sparse read cost |
|------|-------------------|----------------|--------|--------|--------|------------------|
| 4K   | ~75%              | 1.97           | 1.95   | 1.96   | 1.96   | −5% vs full      |
| 8K   | 37%               | 1.18           | 1.19   | 1.20   | 1.18   | −16%             |
| 16K  | 18.75%            | 1.244          | 1.198  | 1.221  | 1.208  | −33% (4.2 vs 6.3ms) |

- **Headline: the trained multi-step drafter barely needs the middle of
  the trunk cache.** Sparse read is FREE at 4K/8K; at 16K it holds
  96.3% of full acceptance while reading ~1/5 of the cache at 2/3 the
  cost. The full-read cost grows with context; the sparse read is FLAT
  (4.2ms at every tier) — the gap widens exactly as the delta scaling
  story predicts.
- 16K honest caveat: paired per-prompt diffs show a small consistent
  full edge (4/6 positive, 2 ties, 0 negative) — say "~96% of full at
  ~1/5 visibility", not "indistinguishable". delta2 anchor-refresh
  recovers about half the gap (98.2% of full).
- **32K+ is genuinely blocked on 1×H100** for the 31B model (fit
  ladder: 16K peaks 75.8GB; three strategies exhausted — offload is
  upstream-nondeterministic, chunked prefill uncertifiable, single-shot
  OOMs). This is the natural handoff to bigger internal compute — it
  SUPPORTS the manager pitch rather than weakening it.

## 2. MiMo-7B production MTP head × delta (ran 07-21, never sent)
Stats: §R. Xiaomi's MiMo-7B-RL ships a DeepSeek-style depth-1 MTP head.

- Zero-shot delta swap inside the production-trained head (no
  retraining): **free within noise** — |Δ| ≤ 1.7pts acceptance across
  all 8 (suite × length) cells, sign flips both directions, mean +0.1.
- Acceptance vs context (dense head): gentle decay, no collapse
  (qa 0.714→0.665, gov 0.588→0.543 to ~12K real tokens; LongBench docs
  cap the measurable range there).
- Together with Gemma: TWO independently-trained production MTP heads
  tolerate delta reads zero-shot.

## 3. MTP mechanism probe, Llama testbed (ran 07-21, never sent)
Stats: §P. DeepSeek/MCore-style 1-layer MTP module trained from scratch
on a frozen Llama-3.1-8B trunk, module attention the only variable.

- The cheap training recipe does NOT produce a usable drafter (acc 0.21
  — a head-quality result, not a delta result).
- BUT delta≈dense INSIDE the module wherever measured (losses 3.66 vs
  3.71; acceptance within 1–2pts) — the same invariance the two
  production heads then confirmed at scale.

## 4. Supporting additions since the 1:03 AM message
- Full inference-latency curve behind the "47×" number now has every
  intermediate point + repro stability (§T3): dense 432.5/1761/7128/
  29005 ms per layer at 131K/262K/524K/1M vs delta-flexrow 30.3/78.8/
  202.6/612.5 → 14.3/22.3/35.2/47.4×; the 1M delta point is
  dual-confirmed by two implementations 2% apart; 1M dense reproduces
  across processes within 0.5%.
- flexrow-vs-masked anchor numerical A/B cert is wired into the bench
  (runs automatically next anchorbench launch); until then the claim is
  "same computation by construction, chunking verified exactly".
- Ready-to-run (user-approved, not yet run): full 16-task English
  LongBench (official files vendored verbatim, commit-pinned) + RULER
  on the 32K-trained arms — one ~2h box (`evalgaps` chain).
- Methodology: 5 high-effort adversarial review rounds on this stretch;
  all confirmed findings fixed before numbers were recorded as final
  (notable catches: prefill/decode conflation in G0 timing, cache-type
  asymmetry, a vacuous output-match metric, deltaN cadence off-by-one).

## Suggested narrative for the Jeff message
1. MTP experiment is done, on TWO production heads (Gemma 4 = trained
   multi-step drafter; MiMo = deployed depth-1 head): delta reads are
   free-to-~96% zero-shot, and the drafter read-cost saving grows with
   context (flat 4.2ms vs growing full-read cost).
2. The G1 table above is the quotable artifact; gates are
   zero-tolerance and everything is run-ID'd.
3. 32K+ needs more than one 80GB card for the 31B trunk — a concrete,
   bounded ask that fits the "bigger experiments" resourcing pitch.
