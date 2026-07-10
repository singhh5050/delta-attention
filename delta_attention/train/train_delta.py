"""WP-2 Task B: LoRA fine-tuning with the delta pipeline active (smoke-first).

Smoke (docs/WP2 acceptance): 50 steps, seq_len 8192, 1 GPU, CE loss, all
metrics logging. This IS the real trainer at small dials — no separate smoke
code path.

    python -m delta_attention.train.train_delta --steps 50 --seq-len 8192

Startup validation (first minute, exits 1 on failure): gamma=1 train forward
matches dense SDPA; one forward+backward step is finite; mandatory wandb keys
logged; tokens/s floor after timing steps.

Metrics per step: loss, lr, tokens_per_sec. Every --probe-every steps:
anchor_grad_ratio (T14 metric; measured 0.379 pre-training — the question is
how it MOVES) and delta_interanchor_cos means on held-out sequences at layers
0/8/16/24 (the drift curve; flattening/rising over training is the headline
claim of WP-2).
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch

PROBE_LAYERS = (0, 8, 16, 24)
MANDATORY_KEYS = ("loss", "ppl", "lr", "tokens_per_sec", "anchor_grad_ratio",
                  "delta_interanchor_cos_mean")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--gamma", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--probe-every", type=int, default=25)
    p.add_argument("--arm", choices=["delta", "dense", "detach"], default="delta",
                   help="delta: train with the delta pipeline in-graph; "
                        "dense: standard attention (data/optimizer control); "
                        "detach: pipeline active but gradient blocked through "
                        "the reused correction (mechanism control)")
    p.add_argument("--detach-delta", action="store_true", help="same as --arm detach")
    p.add_argument("--min-tokens-per-sec", type=float, default=0.0,
                   help="0 = derive floor from first timing steps * 0.5")
    p.add_argument("--save-dir", type=str, default="checkpoints/smoke")
    return p.parse_args()


def build_model(args):
    from peft import LoraConfig, get_peft_model

    from delta_attention.config import Config
    from delta_attention.sample import init_model

    cfg = Config()
    cfg.attn_implementation = "window"
    cfg.mode = "delta"
    cfg.delta_lambda = args.gamma
    cfg.sliding_window = args.window
    cfg.attn_implementation_original = cfg.attn_implementation
    model, tokenizer = init_model(cfg)
    model.config.delta_lambda = args.gamma
    model.config.sliding_window = args.window
    model.config.log_drift = False
    model.config.detach_delta = args.arm == "detach" or args.detach_delta
    # dense arm trains with standard attention: same data/optimizer/probes,
    # no delta pipeline — isolates "long-text training" from the mechanism
    model.config._attn_implementation = (
        "sdpa" if args.arm == "dense" else "flex_delta_train")
    model.config.use_cache = False

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # from_pretrained returns eval mode; the checkpointing branch in
    # LlamaModel.forward requires self.training — without train() it silently
    # no-ops and all activations stay resident (OOM'd a 40GB A100 at s=4096).
    model.train()
    return model.cuda(), tokenizer


def packed_pg19(tokenizer, seq_len, seed=0, max_chunks_per_doc=4):
    """Stream PG19 (shuffled) and yield packed [1, seq_len] id tensors.

    Shuffle + a per-document chunk cap force corpus diversity: without them
    the smoke run trained 50/50 steps on PG19 document #1 (the King James
    Bible, 1.15M tokens — memorized, loss floor ~0.06, zero signal).
    """
    from datasets import load_dataset

    # deepmind/pg19 is a legacy script dataset (unsupported by modern
    # `datasets`); emozilla/pg19 is the standard parquet mirror of the same
    # corpus (verified present on the Hub).
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=64)
    for doc in ds:
        toks = tokenizer.encode(doc["text"], add_special_tokens=False)
        for c in range(min(len(toks) // seq_len, max_chunks_per_doc)):
            yield torch.tensor([toks[c * seq_len:(c + 1) * seq_len]], dtype=torch.long)


def probe_drift(model, batch, gamma, window):
    """delta_interanchor_cos at PROBE_LAYERS via the flex ops on a held-out
    batch: hook layer inputs, recompute q/k/v, measure consecutive anchor
    delta cosines. No hip needed; no_grad."""
    import torch.nn.functional as F

    from delta_attention.llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
    from delta_attention.train.flex_delta import _get_flex, anchor_layout, get_block_mask

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = base.model.layers
    captured = {}

    def hook(idx):
        def fn(mod, inp, out):
            captured[idx] = inp[0].detach()
        return fn

    handles = [layers[i].input_layernorm.register_forward_hook(hook(i)) for i in PROBE_LAYERS]
    with torch.no_grad():
        base(batch.cuda(), use_cache=False)
    for h in handles:
        h.remove()

    means = {}
    with torch.no_grad():
        for i in PROBE_LAYERS:
            attn: LlamaAttention = layers[i].self_attn
            hidden = layers[i].input_layernorm(captured[i])
            hs = hidden.shape[:-1]
            q = attn.q_proj(hidden).view(*hs, -1, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(hidden).view(*hs, -1, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(hidden).view(*hs, -1, attn.head_dim).transpose(1, 2)
            s = q.size(2)
            pos = torch.arange(s, device=q.device).unsqueeze(0)
            cos, sin = base.model.rotary_emb(v.transpose(1, 2), pos)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k, v = repeat_kv(k, attn.num_key_value_groups), repeat_kv(v, attn.num_key_value_groups)
            scale = attn.head_dim ** -0.5
            # MUST be the compiled wrapper: uncompiled flex_attention falls
            # back to eager and materializes the full [h, s, s] score matrix
            # (~8.6 GB/layer at s=8192) — guaranteed OOM mid-probe.
            bm = get_block_mask(s, window, 1024, str(q.device))
            sparse = _get_flex()(q, k, v, block_mask=bm, scale=scale).transpose(1, 2)
            idx, _, s_p = anchor_layout(s, gamma)
            idx = idx.to(q.device)
            key_pos = torch.arange(s, device=q.device)
            mask = (key_pos.unsqueeze(0) <= idx.unsqueeze(1)).view(1, 1, idx.numel(), s)
            dense = F.scaled_dot_product_attention(
                q[:, :, idx], k, v, attn_mask=mask, scale=scale).transpose(1, 2)
            deltas = dense - sparse[:, idx]
            c = F.cosine_similarity(deltas[:, :-1].float(), deltas[:, 1:].float(), dim=-1)
            means[i] = c.mean().item()
    return means


def probe_anchor_grad_ratio(model, batch, gamma, window, detach_delta=False):
    """anchor_grad_ratio on MODEL-DERIVED q/k/v (layer 0 of the current
    weights, real data). The smoke-run version used fixed random tensors,
    which is weights-independent and therefore constant across training —
    useless as a trajectory metric (0.407 at every probe).

    detach_delta must mirror the arm's training forward: without it the
    probe always measures the full-graph ratio, so detach-arm logs say
    nothing about that arm's actual gradient path (pilot runs atyhqiir/
    2hepajmg both logged the full-graph value)."""
    from delta_attention.llama import apply_rotary_pos_emb
    from delta_attention.train.flex_delta import anchor_grad_ratio, delta_forward_train

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layer = base.model.layers[0]
    with torch.no_grad():
        hidden = base.model.embed_tokens(batch.cuda())
        hidden = layer.input_layernorm(hidden)
        attn = layer.self_attn
        hs = hidden.shape[:-1]
        q = attn.q_proj(hidden).view(*hs, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(hidden).view(*hs, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(hidden).view(*hs, -1, attn.head_dim).transpose(1, 2)
        pos = torch.arange(q.size(2), device=q.device).unsqueeze(0)
        cos, sin = base.model.rotary_emb(v.transpose(1, 2), pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    q = q.detach().requires_grad_(True)
    out = delta_forward_train(q, k.detach(), v.detach(), gamma=gamma, window=window,
                              detach_delta=detach_delta)
    out.float().pow(2).mean().backward()
    return anchor_grad_ratio(q.grad, gamma)


def startup_validation(model, args, run):
    from delta_attention.train.flex_delta import delta_forward_train

    def fail(step, reason):
        print(f"[train startup_validation] FAIL at {step}: {reason}", flush=True)
        sys.exit(1)

    # gamma=1 => dense (T13 quick form, self-contained)
    torch.manual_seed(0)
    q = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        train_out = delta_forward_train(q, k, v, gamma=1, window=args.window)
        kk = k.repeat_interleave(4, dim=1)
        vv = v.repeat_interleave(4, dim=1)
        dense = torch.nn.functional.scaled_dot_product_attention(
            q, kk, vv, is_causal=True, scale=128 ** -0.5).transpose(1, 2)
    cos = torch.nn.functional.cosine_similarity(
        train_out.flatten(2).float(), dense.flatten(2).float(), dim=-1)
    if cos.mean().item() <= 0.999:
        fail("gamma1-dense", f"mean cos {cos.mean().item():.6f}")
    run.log({k: 0 for k in MANDATORY_KEYS})
    print("[train startup_validation] PASS (gamma=1 dense check, wandb keys)", flush=True)


def main():
    args = parse_args()
    import os

    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"wp2_{args.arm}_s{args.seq_len}_n{args.steps}",
                     config=vars(args))

    model, tokenizer = build_model(args)
    startup_validation(model, args, run)

    data = packed_pg19(tokenizer, args.seq_len)
    heldout = [next(data) for _ in range(2)]  # drift probe sequences

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr_min)

    floor = args.min_tokens_per_sec
    t_hist = []
    for step in range(1, args.steps + 1):
        batch = next(data).cuda()
        t0 = time.monotonic()
        out = model(input_ids=batch, labels=batch)
        loss = out.loss
        if not torch.isfinite(loss):
            print(f"[train] FATAL: non-finite loss at step {step}", flush=True)
            sys.exit(1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        dt = time.monotonic() - t0
        t_hist.append(dt)
        tps = args.seq_len / dt

        log = {"loss": loss.item(), "ppl": math.exp(min(loss.item(), 20.0)),
               "lr": sched.get_last_lr()[0], "tokens_per_sec": tps, "step": step}
        if step % args.probe_every == 0 or step == args.steps:
            log["anchor_grad_ratio"] = probe_anchor_grad_ratio(
                model, heldout[0], args.gamma, args.window,
                detach_delta=model.config.detach_delta)
            # decomposition for the delta arm: how much of the ratio is the
            # gamma-fold summed correction vs the anchor's key-dense forward
            if not model.config.detach_delta:
                log["anchor_grad_ratio_detached"] = probe_anchor_grad_ratio(
                    model, heldout[0], args.gamma, args.window, detach_delta=True)
            drift = {}
            for hb in heldout:
                for layer, m in probe_drift(model, hb, args.gamma, args.window).items():
                    drift.setdefault(layer, []).append(m)
            for layer, vals in drift.items():
                log[f"delta_interanchor_cos/layer_{layer:02d}"] = sum(vals) / len(vals)
            log["delta_interanchor_cos_mean"] = sum(
                sum(v) / len(v) for v in drift.values()) / len(drift)
            print(f"[train] step {step}: loss {loss.item():.4f} "
                  f"drift_mean {log['delta_interanchor_cos_mean']:.4f} "
                  f"agr {log['anchor_grad_ratio']:.3f}", flush=True)
        run.log(log)

        if step == 10:
            measured = args.seq_len / (sum(t_hist[5:]) / len(t_hist[5:]))
            floor = floor or measured * 0.5
            if measured < floor:
                print(f"[train] FATAL: {measured:.0f} tokens/s under floor {floor:.0f} "
                      "— wiring mistake (fp32? no flash path?)", flush=True)
                sys.exit(1)
            print(f"[train] tokens/s {measured:.0f} (floor {floor:.0f})", flush=True)

    model.save_pretrained(args.save_dir)
    # boxes self-terminate on completion — the adapter must outlive the disk.
    # LoRA r16 on q/k/v/o is ~50MB; wandb artifact is the durable home.
    artifact = wandb.Artifact(f"wp2_adapter_{args.arm}", type="lora-adapter",
                              metadata={"steps": args.steps, "seq_len": args.seq_len,
                                        "gamma": args.gamma, "arm": args.arm})
    artifact.add_dir(args.save_dir)
    run.log_artifact(artifact)
    run.summary["final_loss"] = loss.item()
    run.finish()  # blocks until artifact upload completes
    print(f"[train] DONE: {args.steps} steps, final loss {loss.item():.4f}, "
          f"adapter saved to {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
