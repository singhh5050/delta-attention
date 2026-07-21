# MiMo-7B MTP × delta attention: execution plan (2026-07-21)

Jeff's experiment ("put delta on the MTP transformer layers, see if spec
decoding looks as good as the self-draft results") on a PRODUCTION-trained
head at single-GPU scale. All sources public: XiaomiMiMo/MiMo-7B-RL (HF),
modeling_mimo.py, vLLM mimo_mtp.py, MiMo tech report (2505.07608).

## What the internals say (verified from downloaded code, 07-21)

- **Trunk = Qwen2 architecture exactly** (MiMoForCausalLM subclasses
  Qwen2ForCausalLM): 36 layers, GQA 32q/8kv heads, head_dim 128 (same
  attention geometry as Llama-3.1-8B), RoPE theta 640K, ctx 32,768,
  `use_sliding_window: false` → standard softmax attention everywhere.
  Qwen2-style QKV **bias=True** (the one wiring difference vs Llama).
- **MTP layer** (`model.mtp_layers.0.*`, 16 tensors, ~215M params, present
  in BOTH Base and RL checkpoints — tuned in pretraining+SFT, frozen in
  RL): token_layernorm(emb) ⊕ hidden_layernorm(trunk_hidden) →
  input_proj(2d→d) → ONE standard Qwen2 decoder layer → final_layernorm →
  shared lm_head. Structurally identical to our Track A MTPModule (same
  concat order: [hidden; emb]).
- **Inference wiring (authoritative, from vLLM mimo_mtp.py):** the layer
  consumes the trunk's POST-final-norm hidden states; it keeps ITS OWN KV
  cache; positions are absolute (shared rope); `inputs_embeds[positions==0]
  = 0` (position-0 masking quirk — replicate); and vLLM asserts
  `spec_step_idx == 0`: **only K=1 is natively supported** — deeper
  chaining is our exploratory addition, labeled as such.
- Published acceptance: ~90% with one MTP layer (tech report; presumably
  short-context chat).

## Adaptation (builds on Track A; trunk needs NO port)

- Trunk loads as plain `Qwen2ForCausalLM` with a Qwen2Config coerced from
  MiMo's config (same weight names; `mtp_layers.*` ignored as unexpected
  keys) — no trust_remote_code, no dependence on MiMo's 4.40-era custom
  code under our 4.51.3 pin. Hidden extraction via output_hidden_states
  (hidden_states[-1] = post-norm, matching the vLLM wiring). Trunk runs
  DENSE always (prefill, hiddens, verify) — zero port risk.
- New `delta_attention/mimo_mtp.py`: Track A's MTPModule pattern with
  MiMo's exact math (their four norms, input_proj, QKV bias, silu MLP) and
  weights loaded 1:1 from the safetensors (16-tensor map). Dense vs delta
  reads on the module's own cache reuse the Track A v2 anchor-corrected
  state machine unchanged.
- Eval = mtp_eval.py with a MiMo build_trunk (same accept_block, parity
  gates, per-position logging; chat prompts via MiMo's own tokenizer
  template).

## Phases & gates

- **M0 — wiring calibration (~0.5 day code + $2):** K=1 acceptance at
  short context (2–4K) must reach ≥0.8 (their claim ~0.9). Below that =
  OUR wiring is wrong (pre/post-norm hidden, template, pos-0 mask) — fix
  before interpreting anything. This gate converts their published number
  into a free harness certification.
- **M1 — acceptance vs context ($5):** native dense head, K=1, same
  documents truncated to 4K/8K/16K/32K tiers (truncate_middle), GovReport
  + QA. Decay or flat — either is a result; decay = the production
  motivation measured at 7B.
- **M2 — zero-shot delta swap ($3):** sink+window+anchor-corrected reads
  on the head's cache, head weights untouched. How much acceptance
  survives without any retraining?
- **M3 — delta head finetune ($5–10):** head-only finetune FROM their
  trained weights with delta attention in-graph (Track A trainer;
  chat-mix + long-doc data this time), re-measure the M1 grid.
  Deliverable: acceptance-vs-context curves — native dense vs delta
  (zero-shot) vs delta (finetuned) — Jeff's question answered on a
  production head.

Total: ~1 day of harness work + ~1 day of runs, ≤$25, one H100, entirely
within the Lambda cap. The 512K version of this story still requires
Nemotron Super (MiMo caps at 32K) — this is the mechanism proof on a real
head, not the very-long-context result.

## Known risks

1. Hidden-state tap point wrong → M0 catches (can't reproduce 0.9).
2. Their 90% is chat-domain; our suites may sit lower even with correct
   wiring — M0 uses chat-style short prompts to separate wiring from
   domain.
3. K>1 chaining has no native reference (vLLM supports K=1 only) — report
   K=1 as the headline, chaining as exploratory.
4. QKV bias + rope theta 640K are the two places a silent numerical
   mismatch could hide — both exercised directly by M0.
