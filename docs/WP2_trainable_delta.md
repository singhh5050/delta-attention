# WP-2 — Fine-tuning With the Delta Pipeline Active

You are a coding agent in the delta-attention fork. Prereq: WP-0 merged (Gates 1–2 green). Read `00_MASTER_PLAN.md`; rules binding. This WP builds a DIFFERENTIABLE delta forward, a training loop, and post-training evaluation hooks. Runs on a dedicated instance, never co-tenanted with eval jobs.

## Why a new path is needed

The inference path is non-differentiable: `qsa_kernel` (delta_kernel.py) is a forward-only Triton kernel and the sparse side uses hip-attn inference kernels. Training requires gradients through BOTH the sparse output and the anchor dense rows, so the delta (dense−sparse, computed at anchors, broadcast to the block) is inside autograd — gradient from all γ positions flows back into each anchor's computation. That gradient concentration is the scientific object of this WP.

## Task A — differentiable delta forward (FlexAttention)

New module `delta_attention/train/flex_delta.py` with `delta_forward_train(q, k, v, gamma, window, sink)`:
- Sparse side: `torch.nn.attention.flex_attention` with a block mask = causal ∧ (col < sink ∨ row − col < window). Compile the block mask once per (s, window, sink).
- Query-sparse side: gather anchor rows (uniform γ; include the dense tail block replicating `cut_n` semantics) and run flex_attention (or plain SDPA — anchor count is small) with full causal masking for those rows.
- Delta = dense_anchor − sparse[anchor rows]; broadcast within blocks (reuse WP-1's `apply_delta_variable` if merged, else the uniform reshape); return corrected output.
- Everything in the graph; no `.detach()` anywhere on the delta path (a config flag `detach_delta=true` exists ONLY as an ablation arm).

**T13 equivalence gate (blocks all training):** on fixed inputs (b=1, s=8192, real model weights, bf16→fp32 compare), `delta_forward_train` vs inference `delta_forward`: mean cos > 0.999, per-row p1 > 0.995, and γ=1 ⇒ matches dense (T3 analog). Document any principled mismatch (e.g., hip-attn window semantics vs flex mask) and reconcile the MASK, not the tolerance.
**T14 gradient sanity:** grads flow to q/k/v projections from a scalar loss; gradient norm at anchor-row positions vs non-anchor positions logged (expect anchor concentration — this ratio, `anchor_grad_ratio`, is a mandatory training metric).

## Task B — training loop (train/train_delta.py)

- Model: Llama-3.1-8B-Instruct + LoRA (peft is already in requirements): r=16, α=32, targets q_proj,k_proj,v_proj,o_proj. Full-FT is out of scope for run 1.
- Model surgery: swap LlamaAttention's interface to `delta_forward_train` for all layers during training (γ=64, window=2048, sink=1024 — match inference defaults).
- Data: long-context corpus at `seq_len=32768`. Default: PG19 train split packed to 32K (datasets lib already pinned). Config hook for a second mixture later; do not add one now.
- Losses (separate configs, not mixed in run 1):
  - `loss=ce`: next-token cross-entropy with pipeline active.
  - `loss=distill`: MSE between per-layer attention outputs (post-correction) and a frozen full-attention teacher's outputs (same base weights, FA2), computed on a random 1/8 subset of layers per step to bound memory; plus 0.1·CE anchor term. (This is the "push the corrected vector toward full attention" objective.)
- 8×H100, FSDP or DDP+grad-checkpointing (agent's choice; justify in PR), bf16, cosine LR 1e-4→1e-5, ~500M tokens ≈ 15K steps at 32K×1 per gpu-step. Checkpoint every 1K steps, keep last 3 + best.
- Mandatory wandb: loss, ppl, lr, tokens/s, anchor_grad_ratio, and every 500 steps an in-loop probe: `delta_interanchor_cos` measured on 4 held-out 32K sequences (the drift curve — flattening over training is the headline claim).
- Startup validation (first 60s): T13 quick version at s=4096, one forward+backward step completes, loss finite, all wandb keys logged, tokens/s above floor (config `min_tokens_per_sec`, set from a 20-step timing at launch — under floor means a wiring mistake like accidental fp32 or no flash path → exit(1)).

## Task C — post-training eval hook

Export merged-LoRA checkpoint → serve via the WP-0 server → configs `t2_ce_step{5k,15k}` and `t2_distill_step{5k,15k}` in experiments.yaml run RULER (delta mode γ=64 AND γ=128 — the trained model should tolerate a larger γ if smoothness improved) + drift telemetry. Comparison rows: `base_delta_g64`, `base_delta_g128`.

## Acceptance criteria

- T13, T14 green before any multi-hour run (enforced by startup validation, not by promise).
- One CE training run launched to completion at smoke scale (50 steps, s=8192, 1 GPU) with all metrics logging — this is the WP's smoke test, run before requesting the 8×H100 instance.
- Full run launch = start + validation pass + wandb URL. NO monitoring. Human watches wandb.
- Post-training eval configs runnable through the standard runner with zero special-casing.

## Forbidden

- Mixing losses in run 1; inventing data mixtures; full fine-tuning; touching inference kernels; training at 131K (32K only for run 1); co-tenanting with eval jobs; any polling/monitoring loop.
