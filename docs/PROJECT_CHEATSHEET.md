# Delta Attention — project cheat sheet (as of 2026-07-16)

One page of everything you need cold in a meeting. Every number has a
source; deep detail lives in `STATS.md` (Part II for pre-pivot numbers).

## The mechanism (30-second version)

Delta Attention (Willette et al., NeurIPS 2025) = sparse attention + a
cheap correction. Sparse part: each query attends only to the first 1,024
tokens (**sink**) + the last 2,048 tokens (**sliding window**). Correction:
every **γ = 64**th query row is computed EXACTLY (full dense attention for
that one row); the difference between the exact row and the sparse row
(the "delta") is broadcast to the next 63 rows. During decoding, the same
idea: an exact "anchor" row every γ_dec steps, delta reused in between.

## Setup facts (memorize these five)

| fact | value |
|---|---|
| Model | Llama-3.1-8B-Instruct, bf16, LoRA finetunes (r=16, α=32, on q/k/v/o projections, lr 1e-4→1e-5 cosine) |
| Training data | **PG19** (DeepMind long-books corpus, `emozilla/pg19` mirror), shuffled + packed, ≤4 chunks/document |
| Training budget | 8K arms: 2,000 steps × 8,192 tok; 32K arms: 500 steps × 32,768 tok — both ≈ **16.4M tokens** |
| Sparse geometry | sink 1,024 + window 2,048, correction every γ=64 rows — visible fraction: **37% @8K, 9% @32K, ~3% @128K** |
| Eval protocol | PG19 ppl, 32 held-out chunks @32K (paired per-chunk stats); LongBench v1 QA (official budgets), GovReport ROUGE-L, En.MC, MMLU n=1000, RULER niah |

## Under the hood (what we do and do NOT use)

- Plain **PyTorch + HuggingFace transformers** with custom attention
  kernels (`delta_forward` for sparse+delta, `sdpa_rectangle` for
  block-dense decode/verify). bf16 throughout.
- **NO inference-engine optimizations: no CUDA graphs, no torch.compile,
  no vLLM/SGLang, no paged KV, no fused sampling.** If asked: "naive but
  symmetric PyTorch harness — both arms pay the same overheads, so ratios
  are meaningful; engine-grade absolute numbers are future work."
- Timing protocol (after two corrections, 07-16): warm process, lean
  hand-rolled loop on BOTH sides, cuda.synchronize around every window,
  all 20 prompts timed. History of the headline speedup as biases were
  removed: 1.24–1.37× → 1.20–1.22× → **1.10–1.15× (final)**.

## Key results (one line each, with provenance)

1. **Training gap @32K**: delta-trained beats dense-trained under the
   sparse pipeline by −0.025 ppl, 32/32 chunks, 3 seeds (runs in §A/§N).
2. **"Train cheap, deploy dense"** (Jeff's slide-7 interest): delta-trained
   keeps **96% (8K) / 88% (32K)** of the dense-finetune quality gain under
   dense eval (§A2). Caveat: retained fraction FALLS with length.
3. **Benchmarks unchanged** (LongBench/En.MC tie-dominated, MMLU
   0.638–0.649 all arms ≥ base) = parity at lower cost, not a null (§H/§L).
4. **Self-speculative decoding** (draft = same 8B with delta decode,
   verify = dense rectangle, output exactly dense-greedy): genuine
   acceptance (excluding the structural free first token) base 0.89/0.75/0.53
   at K=2/4/8, ce32k-trained 0.94/0.87/0.75, τ up to 7.26 (§I3).
5. **Wall-clock**: **1.10–1.15× @32K/batch-1** (naive harness). K=8 is
   SLOWER for the base draft (0.86×) — every drafted token costs a full
   forward (~25ms vs dense 36ms), so low-acceptance late positions burn
   more than they save: forwards/token = (K+1)/τ. Trained drafts rescue
   K=8 (1.04×). Cost model reproduces every measured cell within ~3%.
6. **RULER negative control**: sparse-draft acceptance collapses (0.15–0.27
   genuine) on needle retrieval, parity byte-identical — the draft can't
   know what it can't see; confirms the mechanism honestly (§I3).
7. **Gradient story** (Jeff's 1/γ question): anchor gradients are ~3.4–4.6×
   larger but damping them HURTS monotonically (interventional, §K), and
   the learned LoRA update tilts toward neither anchor nor window gradient
   (cos at the 1/√d chance floor — probe has power: g_A·g_W is 100–400×
   chance) (§M).

## Positioning vs DeepSeek Sparse Attention (DSA)

The meeting answer: **complementary, not competing — DSA picks better
sparse patterns; delta corrects whatever pattern you picked.**

- DSA = a learned "lightning indexer" selects top-k keys per query
  (better WHICH-keys selection), trained in from scratch/continued
  pretraining. Our window+sink is the dumbest possible pattern — the
  delta correction is pattern-agnostic (anchor rows are exact dense rows
  no matter what mask made the other rows), so it could sit ON TOP of a
  DSA-style indexer: DSA reduces what the sparse rows miss, delta
  periodically corrects what's still missed.
- Different axis: DSA sparsifies keys-per-query; delta sparsifies
  exact-rows-per-query (periodic exact rows + broadcast correction).
  Composable by construction.
- Our training result targets a different user: DSA needs
  (continued-)pretraining at scale; delta-training is a **retrofit path
  for existing dense checkpoints** (Llama etc.) — 500 LoRA steps, no
  architecture change, 88–96% of dense-finetune quality.
- The self-spec/MTP drafting angle applies to a DSA model too: any sparse
  decoder + periodic dense anchoring gives a draft whose output a dense
  verify can certify.

## Corrections owed / known caveats

- We do NOT use CUDA graphs (misstatement in the 07-16 meeting — correct
  proactively).
- Speedup numbers are naive-harness ratios; engine-grade measurement is
  open work.
- "Needle span is where RULER acceptance dies" is inferred from positional
  curves, not token-level logged.
- QA parity certification covers ~15 tokens × 3 prompts per cell (eos is
  early); GovReport prefixes (45–436 tokens) are the strong certification.
