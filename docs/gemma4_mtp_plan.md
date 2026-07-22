# Gemma 4 MTP drafter × delta attention: scoping (2026-07-22)

Verified from primary sources (configs + transformers-main modeling code,
downloaded to ~/Development/delta-attention/gemma4-scoping/):
ai.google.dev/gemma/docs/mtp/mtp, google/gemma-4-{31B,26B-A4B}-it(-assistant)
configs, transformers models/gemma4_assistant/modeling_gemma4_assistant.py.

## What Gemma 4's MTP actually is (all verified, no inference)

- Every Gemma 4 variant ships a PAIRED, jointly-trained "assistant" drafter
  (repo = target id + "-assistant"). Invoked via vanilla HF assisted
  decoding: `target.generate(..., assistant_model=assistant)`. Drafts
  MULTIPLE tokens per verify round with a heuristic count scheduler —
  trained multi-step chaining, up to ~3x measured speedup (Google blog).
- Drafter architecture (31B assistant config): 4 layers [3x sliding
  (window 1024) + 1x full_attention], dim 1024, 262K max positions,
  centroid-clustered cheap head (2048 clusters, top-32) with a full
  lm_head fallback flag.
- **The decisive wiring (modeling code): the drafter keeps NO KV cache of
  its own** (`use_cache=False`). It receives `shared_kv_states` — the
  TRUNK's cached K/V for the last layer of each layer type — and its
  layers CROSS-ATTEND to those as encoder states (bidirectional masks
  over the trunk cache). Input = pre_projection(2*backbone_hidden -> 1024)
  of concatenated trunk-state/embedding features; post_projection returns
  a backbone-sized hidden for the next chained step.

## Why this is the ideal delta target

At long context, the drafter's full_attention layer cross-reads the
trunk's ENTIRE full-attention KV cache per drafted token (262K positions
max). That read is precisely what delta attention cheapens: sink + window
+ periodic exact rows over a large KV. No module cache to manage (unlike
MiMo), no varlen repacking — delta-ification = subsampling the
shared_kv_states read, with our WP-3 anchor-correction state machine on
top. The sliding layers already read window-limited trunk KV (nothing to
do there — Gemma's own design agrees locality is enough for 3/4 layers,
which is itself corroborating evidence for the delta thesis).

## Feasibility facts

- Single GPU: 31B dense bf16 ~62GB, 26B-A4B ~52GB -> one 80GB H100.
- Trunk = interleaved sliding/full softmax attention (5:1) — standard
  math, runs vanilla; we never port it.
- transformers: gemma4 + gemma4_assistant are in MAIN (2026) — NOT in our
  pinned 4.51.3. The Gemma harness runs in its OWN venv on the box
  (isolated from the hip-attn stack; our flex delta op only needs torch).
- Gemma weights are license-gated on HF — accept the Gemma 4 license with
  the project token BEFORE launching anything (a box 401 is a silly way
  to discover this).

## Phased plan (mirrors the MiMo playbook)

- **G0 — driver + baseline (1 day code, ~$5):** clean-venv harness: load
  target + assistant, reproduce assisted decoding via the vanilla API,
  measure native acceptance/token-counts per round with our accept_block
  protocol at 4K/32K/128K/262K tiers (long docs: InfiniteBench books —
  also fills the >12K gap MiMo couldn't reach). GATE: our measured
  speedup/acceptance is consistent with the published ~3x at short ctx.
- **G1 — the read-path swap (~2 days code):** reimplement the drafter
  forward explicitly (4 layers, our style — weights loaded 1:1, masks
  reproduced from create_bidirectional_* semantics), verify G0 parity,
  then delta-ify the full_attention cross-read (sink+window+anchor over
  the trunk KV). Measure acceptance dense-read vs delta-read across
  tiers; zero-shot first, short adapter finetune only if needed.
- **G2 — cost curve:** headread-style latency for the drafter's full
  layer at 32K->262K cache (extends the anchorbench headread cells with
  Gemma dims). Deliverable: acceptance parity + read-cost curve = the
  trained-multi-step version of the MiMo result, single GPU.

## Risks
1. The assisted-decoding driver's exact inputs_embeds recipe (what gets
   concatenated) lives in the generation-side candidate generator — read
   it before G1 (same class of risk as MiMo's hidden-tap, and G0's parity
   gate catches a mistake).
2. Bidirectional-mask semantics in the drafter (the flip tricks in
   create_attention_masks) must be reproduced exactly in G1.
3. New-transformers venv on boxes = new setup path (isolated; does not
   touch the pinned production env).
4. Centroid head: replicate or disable via use_ordered_embeddings — G0
   measures which the shipped checkpoints use.

## Weight layout (verified 07-22 via safetensors header of
## google/gemma-4-31b-it-assistant — single file, 41 tensors, 469.5M params;
## repo NOT gated for our token, so no license blocker)

- `model.embed_tokens.weight [262144, 1024]`, lm_head TIED. **No centroid /
  masked_embedding tensors in this checkpoint** -> the 31B assistant ships
  with the plain lm_head path (use_ordered_embeddings off); risk 4 closed.
- **No k_proj / v_proj anywhere.** Each layer has only q_proj + q_norm +
  o_proj: the drafter is PURE CROSS-ATTENTION over the trunk's cached KV
  (num_kv_shared_layers=4 means all four layers borrow trunk KV; drafted
  tokens see each other only through the chained inputs_embeds). This is
  the strongest possible version of the delta fit: delta-ifying the
  drafter = subsampling a read over someone else's cache — there is no
  drafter-side KV state to keep consistent at all.
- Layers 0-2 (sliding, window 1024): q_proj [8192,1024] = 32 heads x 256,
  q_norm [256]. Layer 3 (full attention): q_proj [16384,1024] with q_norm
  [512] — the full-attn layer queries at head_dim 512 to match the trunk's
  full-attention KV layout (confirm exact trunk head mapping in G1).
- `pre_projection [1024, 10752]` = concat of two 5376-dim backbone vectors
  (trunk hidden state (+) embedding path), `post_projection [5376, 1024]`
  returns the chaining hidden; per-layer learned `layer_scalar [1]`.
