# Delta-in-MTP: scoping doc (2026-07-21)

Jeff's direction: replace the transformer layer(s) inside a Nemotron/
DeepSeek-style MTP (multi-token-prediction) stack with delta attention,
and evaluate speculative decoding against our full-model self-draft
baseline (§I3 of `stats_2026-07-15.md`). This doc pins the design and the
go/no-go gates BEFORE GPU spend.

## 1. Why this is the right vehicle (from our own data)

Self-speculation with the full 8B as draft measures forwards-per-emitted-
token = (K+1)/τ — every drafted token pays a full weight read, so measured
wall-clock capped at 1.10–1.15× @32K and K=8 was a slowdown (0.86×). But
the delta draft's ACCEPTANCE is excellent at long context (genuine 0.94/
0.87/0.75 at K=2/4/8 for the trained draft; τ up to 7.26).

An MTP head drafts a token for ~1 layer ≈ 1/32 of a forward:
forwards/token ≈ (1 + K/32)/τ. At our measured τ=7.26 (K=8) that is 0.17
forwards/token → ~5–6× fewer trunk forwards, EAGLE-class, IF a 1-layer
head can approach the full-model draft's acceptance. That transfer is the
experimental question. Delta attention's role: keep the head's attention
long-context-capable (the acceptance driver at 32K) at sparse cost.

## 2. Architecture decisions (proposed)

- **Head style: DeepSeek-V3 MTP module** (what Nemotron-style stacks
  follow): RMSNorm(h_t) ⊕ RMSNorm(emb(x_{t+1})) → linear projection to
  d_model → ONE transformer layer → shared frozen lm_head. Trunk (all 32
  Llama layers) FROZEN; only the module trains (~220M params at 8B scale).
- **Attention inside the module**: the module keeps its OWN KV cache over
  its own hidden sequence (DeepSeek semantics). Variant A (baseline):
  full dense attention in the module. Variant B (the idea): delta
  attention in the module — sink 1024 + window 2048 + exact anchor row
  every γ (prefill), per-block anchor at decode, i.e. exactly our
  existing kernel dropped into one layer.
- **Depth**: start with ONE module (predicts t+2 → drafts 1 extra token
  per trunk forward, K_eff=2). Chained depth (K=4/8) only after the
  single-module gate passes. (EAGLE-style tree drafting explicitly out of
  scope for the probe.)
- **Base model**: Llama-3.1-8B-Instruct (all our baselines/adapters/
  harness live there). No public Llama-8B MTP checkpoint exists —
  DeepSeek-V3's MTP weights are welded to a 671B MoE — so we train our
  own. OPEN QUESTION (Harsh/NVIDIA): internal Nemotron MTP recipes or
  Llama-class MTP checkpoints would shortcut Phase A — worth asking.

## 3. Training recipe (Phase A)

- Data: PG19 + a chat/instruct mix (the draft must accept on QA/summary
  text, not just books); 8K seqs, ~2000 steps, lr 1e-4 → the same
  budget class as our LoRA pilots. Loss: CE of the module's t+2
  prediction against ground truth (teacher-forced), optionally + KL to
  the trunk's own next-token distribution shifted by one (distill flavor
  — decide after CE-only baseline).
- Cost: trunk frozen → fits ONE H100 at 8K comfortably (trunk fwd
  no-grad + 1 trainable layer); est. 2–4h/run. 32K finetune pass after
  the 8K gate: +2–4h.
- Deliverable per run: module checkpoint + acceptance eval (below).

## 4. Evaluation (reuses the certified spec-dec harness)

`specdec_eval.py` machinery carries over: accept_block (prefix
acceptance, offline-tested), dense-verify rectangles on the shared cache,
parity gate v5, per-position/genuine acceptance logging, GovReport + QA +
RULER-negative-control suites, warm lean-loop timing. New code is ONLY
the draft-proposal generator (MTP module forward instead of full-model
delta decode) — est. 1–2 days of harness work, most of it cache
plumbing for the module's KV.

Metrics, same definitions as §I3: genuine (pos-2+) acceptance, τ,
forwards-per-token (now (1 + K/32)/τ), wall-clock vs the lean dense
baseline @32K batch-1. Baselines to beat: (a) full-model ce32k delta
draft (genuine 0.87 @K=4, τ 4.6; 1.13× wall-clock), (b) module with
DENSE attention (isolates what delta contributes inside the head).

## 5. Phases & kill criteria

- **Phase 0 (this doc)**: design + open questions. DONE.
- **Phase A (~2 box-days)**: single dense-attention MTP module on frozen
  Llama-8B; measure acceptance @32K on GovReport/QA.
  GATE: module acceptance at pos-2 ≥ 0.5 on GovReport (else a 1-layer
  head can't draft this model at long context and the direction dies
  cheaply here).
- **Phase B (~1 box-day)**: swap delta attention into the module (Variant
  B), retrain, same eval. GATE: delta-module acceptance within ~5pts of
  dense-module at 32K while cutting module attention cost; check 64K
  where dense-module attention gets expensive/degraded — this is where
  delta should WIN, not just tie.
- **Phase C (~1–2 box-days)**: chain to K=4/8, wall-clock grid, RULER
  negative control, write-up vs §I3 table.

Total estimate: 4–6 box-days GPU + ~2–3 days harness/integration work,
AFTER the triad is locked (it is, as of §O4). Risks: (1) 1-layer head
acceptance at 32K is unknown — hence Phase A gate; (2) no reference
implementation to check against — mitigated by the parity gate (output
must equal dense greedy by construction, same certification as §I3);
(3) tokenizer/template quirks if we swap base models later — keep
Llama-3.1-8B until Phase C.

## 6. What can be told to Jeff today

Design + phase plan + the forwards-per-token argument (§1). NO numbers
exist yet; first numbers land after Phase A trains. The honest one-liner:
"self-spec proved the acceptance is there and the cost structure isn't;
MTP fixes the cost structure; the experiment is whether a 1-layer delta
head keeps the acceptance."
