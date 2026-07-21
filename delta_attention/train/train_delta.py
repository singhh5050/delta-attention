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
from contextlib import contextmanager

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
    p.add_argument("--arm", choices=["delta", "dense", "detach", "distill"], default="delta",
                   help="delta: train with the delta pipeline in-graph; "
                        "dense: standard attention (data/optimizer control); "
                        "detach: pipeline active but gradient blocked through "
                        "the reused correction (mechanism control); "
                        "distill: delta pipeline forward, loss = KL to the frozen "
                        "dense teacher (adapters disabled) instead of CE")
    p.add_argument("--detach-delta", action="store_true", help="same as --arm detach")
    p.add_argument("--delta-grad-scale", type=float, default=1.0,
                   help="backward-only multiplier on the broadcast correction's "
                        "gradient (delta arm; Jeff's 1/gamma idea: try 0.125 = "
                        "1/sqrt(64) and 0.015625 = 1/64). 1.0 = unchanged")
    p.add_argument("--data-seed", type=int, default=0,
                   help="PG19 shuffle seed (seed replication runs)")
    p.add_argument("--model", type=str, default="",
                   help="HF model id override (must be Llama-architecture; "
                        "empty = Config default Llama-3.1-8B-Instruct)")
    p.add_argument("--data-source", choices=["pg19", "arxiv"], default="pg19",
                   help="pg19: long books, per-doc 32K chunks (original). "
                        "arxiv: LaTeX papers packed ACROSS documents (papers "
                        "average ~10K tokens, shorter than one 32K sequence)")
    p.add_argument("--bench", action="store_true",
                   help="training-efficiency benchmark: CUDA-synced fwd/bwd/"
                        "step timing + peak memory over the timed steps "
                        "(after --bench-warmup), written to "
                        "results/trainbench.csv. Same loop as real training "
                        "— symmetric across arms by construction")
    p.add_argument("--dense-impl", choices=["sdpa", "flash_attention_2"],
                   default="sdpa",
                   help="attention kernel for the dense arm (sdpa = "
                        "historical default; fa2 = what every eval-side "
                        "dense path uses — bench BOTH to separate kernel "
                        "choice from mechanism)")
    p.add_argument("--bench-warmup", type=int, default=5,
                   help="untimed steps before timing starts (kernel autotune "
                        "/ allocator maturation — the same cold-start bias "
                        "that inflated the 07-16 decode speedups)")
    p.add_argument("--distill-alpha", type=float, default=0.0,
                   help="CE weight mixed into the distill loss (0 = pure KL)")
    p.add_argument("--teacher-checkpoint", type=str, default="",
                   help="distill arm only: LoRA adapter merged into a SEPARATE "
                        "frozen dense teacher model (e.g. checkpoints/pilot_dense "
                        "-> distill onto the dense-finetuned model). Default '' "
                        "keeps the same-model base teacher (adapters disabled). "
                        "The student still starts from the plain base, so saved "
                        "adapters eval exactly like every other arm's")
    p.add_argument("--tag", type=str, default="",
                   help="suffix for the wandb run/artifact names (e.g. _32k) so "
                        "re-trainings don't clobber wp2_adapter_<arm>:latest")
    p.add_argument("--no-artifact", action="store_true",
                   help="skip the wandb adapter upload (smoke runs — keeps "
                        "throwaway adapters out of the wp2_adapter_* namespace "
                        "that fetch_adapters treats as the source of truth)")
    p.add_argument("--min-tokens-per-sec", type=float, default=0.0,
                   help="0 = derive floor from first timing steps * 0.5")
    p.add_argument("--save-dir", type=str, default="checkpoints/smoke")
    return p.parse_args()


def build_model(args):
    from peft import LoraConfig, get_peft_model

    from delta_attention.config import Config
    from delta_attention.sample import init_model

    cfg = Config()
    if args.model:
        cfg.model_str = args.model
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
    model.config.delta_grad_scale = args.delta_grad_scale
    # dense arm trains with standard attention: same data/optimizer/probes,
    # no delta pipeline — isolates "long-text training" from the mechanism.
    # --dense-impl controls WHICH dense kernel (07-20 review: sdpa vs
    # flash_attention_2 was an unvalidated confound in the T1 speedup)
    model.config._attn_implementation = (
        args.dense_impl if args.arm == "dense" else "flex_delta_train")
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
    if args.arm != "distill":
        # CE arms take their loss from hidden states (chunked_ce_hidden fuses
        # the lm_head per chunk) — full-vocab logits are never materialized.
        # The distill arm needs real logits for the KL, so it keeps lm_head.
        setattr(model.get_base_model(), "no_lm_head", True)
    return model.cuda(), tokenizer


def build_teacher(checkpoint):
    """Separate frozen dense teacher with a LoRA checkpoint merged in (same
    init_model merge path the eval scripts use). A second 16GB static model is
    the safe way to get a finetuned teacher: PEFT set_adapter switching also
    toggles requires_grad on the student's params — silent no-op training if
    the toggle-back ever misses."""
    from delta_attention.config import Config
    from delta_attention.sample import init_model

    cfg = Config()
    cfg.attn_implementation = "flash_attention_2"
    cfg.mode = "none"
    cfg.attn_implementation_original = cfg.attn_implementation
    cfg.checkpoint = checkpoint
    teacher, _ = init_model(cfg)
    teacher.config.log_drift = False
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.eval().cuda()


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


def packed_arxiv(tokenizer, seq_len, seed=0):
    """Stream arXiv LaTeX papers (shuffled) packed ACROSS documents into
    [1, seq_len] tensors, eos-separated.

    Second-corpus replication (T3): papers average ~10K tokens — shorter
    than one 32K sequence — so unlike PG19's per-doc chunking, sequences
    here concatenate 2-6 papers. Long-range structure differs from books
    by design; that is the point of the replication."""
    from datasets import load_dataset

    # RedPajama/proof-pile-2/scientific_papers are script datasets
    # (unsupported by modern `datasets`); this is a parquet arXiv corpus
    # verified to stream. If it moves, pick any parquet arXiv mirror and
    # keep the packing logic.
    # take() BEFORE shuffle pins training to the first 15,000 CANONICAL
    # docs: streaming shuffle() otherwise permutes the corpus shards, so
    # "training reads the head" was never guaranteed (07-20 review — the
    # 07-17 T3 run used the shuffled-shard stream; its train/eval
    # disjointness was verified post-hoc for that run, not structural).
    # Eval (ppl_eval) reads canonical docs 20,000+ — disjoint BY
    # CONSTRUCTION from here on.
    ds = load_dataset("common-pile/arxiv_papers", split="train",
                      streaming=True)
    ds = ds.take(15000).shuffle(seed=seed, buffer_size=256)
    eos = tokenizer.eos_token_id
    buf = []
    for doc in ds:
        text = doc.get("text") or ""
        if len(text) < 2000:  # skip stubs/withdrawn notices
            continue
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        buf.append(eos)
        while len(buf) >= seq_len:
            yield torch.tensor([buf[:seq_len]], dtype=torch.long)
            buf = buf[seq_len:]


def packed_stream(tokenizer, seq_len, source="pg19", seed=0):
    if source == "arxiv":
        return packed_arxiv(tokenizer, seq_len, seed=seed)
    return packed_pg19(tokenizer, seq_len, seed=seed)


def _chunk_apply(fn, total, *args):
    """One idiom for both loss helpers: gradient-checkpoint the chunk when
    anything requires grad (so no fp32 chunk output is RETAINED for backward
    — plain chunking only bounds temporaries), plain call otherwise (avoids
    checkpoint's no-grad-inputs warning on logging-only paths)."""
    from torch.utils.checkpoint import checkpoint

    if any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
        return total + checkpoint(fn, *args, use_reentrant=False)
    return total + fn(*args)


def _ce_chunk(lg, tg):
    import torch.nn.functional as F

    # reduction="sum" skips ignore_index=-100 positions; callers must
    # normalize by the NON-ignored count, not tg.numel()
    return F.cross_entropy(lg.float().reshape(-1, lg.size(-1)),
                           tg.reshape(-1), reduction="sum")


def _ce_hidden_chunk(h, w, tg):
    import torch.nn.functional as F

    lg = F.linear(h, w)  # lm_head applied per chunk, inside the checkpoint
    return F.cross_entropy(lg.float().reshape(-1, lg.size(-1)),
                           tg.reshape(-1), reduction="sum")


def _kl_chunk(s_lg, t_lg):
    s = torch.log_softmax(s_lg.float(), dim=-1)
    t = torch.log_softmax(t_lg.float(), dim=-1)
    return torch.sum(t.exp() * (t - s))


def chunked_ce(logits, labels, chunk=2048):
    """Shifted CE from logits, fp32 one gradient-checkpointed chunk at a
    time. Used where full-vocab logits already exist (the distill arm needs
    them for the KL). Normalizes by non-ignored (!= -100) label count,
    matching the HF labels= path this replaced. That path was dropped
    because it upcasts full-vocab logits to fp32 in loss_function, and the
    output_logits=False alternative routes through hip's
    memory_efficient_llm_ce — a FORWARD-ONLY Triton kernel (no backward in
    hip-attn 1.2.9), so loss.backward() would raise."""
    lg, tg = logits[:, :-1], labels[:, 1:]
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for i in range(0, lg.size(1), chunk):
        total = _chunk_apply(_ce_chunk, total, lg[:, i:i + chunk], tg[:, i:i + chunk])
    return total / (tg != -100).sum().clamp_min(1)


def chunked_ce_hidden(hidden, lm_head_weight, labels, chunk=2048):
    """Shifted CE straight from hidden states: the lm_head projection runs
    per chunk INSIDE the checkpoint, so full-vocab logits are never
    materialized (~8.4GB bf16 + an equal-size grad at 32K — fits an 80GB
    card but nothing smaller). This is the training loss for the CE arms
    (build_model sets no_lm_head for them)."""
    h, tg = hidden[:, :-1], labels[:, 1:]
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    for i in range(0, h.size(1), chunk):
        total = _chunk_apply(_ce_hidden_chunk, total,
                             h[:, i:i + chunk], lm_head_weight, tg[:, i:i + chunk])
    return total / (tg != -100).sum().clamp_min(1)


def kl_to_teacher(student_logits, teacher_logits, chunk=2048):
    """Per-token mean KL(teacher || student), fp32 one gradient-checkpointed
    sequence-chunk at a time (b=1, like the probes)."""
    total = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    for i in range(0, student_logits.size(1), chunk):
        total = _chunk_apply(_kl_chunk, total, student_logits[:, i:i + chunk],
                             teacher_logits[:, i:i + chunk])
    return total / student_logits.size(1)


@contextmanager
def attn_impl(model, impl):
    """Temporarily flip the attention dispatch; the restore is structural
    (exception-safe), not a statement that only runs on the happy path."""
    prev = model.config._attn_implementation
    model.config._attn_implementation = impl
    try:
        yield
    finally:
        model.config._attn_implementation = prev


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


def probe_anchor_grad_ratio(model, batch, gamma, window, detach_delta=False,
                            delta_grad_scale=1.0):
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
                              detach_delta=detach_delta,
                              delta_grad_scale=delta_grad_scale)
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

    # chunked losses must match their unchunked forms AND backprop (the
    # labels= path they replace upcasts full-vocab logits to fp32 — OOM at
    # 32K — and hip's memory_efficient_llm_ce has no backward). chunk=100
    # over 512 rows exercises multi-chunk + ragged tail; the -100 block
    # exercises ignore-index normalization (divide by NON-ignored count).
    lg = torch.randn(1, 512, 128, device="cuda", requires_grad=True)
    tg = torch.randint(0, 128, (1, 512), device="cuda")
    tg[0, 100:200] = -100
    ref = torch.nn.functional.cross_entropy(
        lg[:, :-1].reshape(-1, 128).float(), tg[:, 1:].reshape(-1),
        ignore_index=-100)
    ours = chunked_ce(lg, tg, chunk=100)
    if abs(ref.item() - ours.item()) > 1e-4:
        fail("chunked-ce", f"full {ref.item():.6f} vs chunked {ours.item():.6f}")
    ours.backward()
    if lg.grad is None or not torch.isfinite(lg.grad).all():
        fail("chunked-ce-grad", "missing/non-finite grad through checkpoint")
    # hidden-CE (lm_head fused per chunk) must match CE over full projection
    hd = torch.randn(1, 512, 64, device="cuda", requires_grad=True)
    w = torch.randn(128, 64, device="cuda")
    ref_h = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hd, w)[:, :-1].reshape(-1, 128).float(),
        tg[:, 1:].reshape(-1), ignore_index=-100)
    ours_h = chunked_ce_hidden(hd, w, tg, chunk=100)
    if abs(ref_h.item() - ours_h.item()) > 1e-4:
        fail("chunked-ce-hidden", f"full {ref_h.item():.6f} vs {ours_h.item():.6f}")
    ours_h.backward()
    if hd.grad is None or not torch.isfinite(hd.grad).all():
        fail("chunked-ce-hidden-grad", "missing/non-finite grad through checkpoint")
    s2 = lg.detach().clone().requires_grad_(True)
    t2 = torch.randn(1, 512, 128, device="cuda")
    sf = torch.log_softmax(s2.detach().float(), dim=-1)
    tf = torch.log_softmax(t2.float(), dim=-1)
    ref_kl = (torch.sum(tf.exp() * (tf - sf)) / s2.size(1)).item()
    kl = kl_to_teacher(s2, t2, chunk=100)
    if abs(ref_kl - kl.item()) > 1e-4:
        fail("chunked-kl", f"full {ref_kl:.6f} vs chunked {kl.item():.6f}")
    kl.backward()
    if s2.grad is None or not torch.isfinite(s2.grad).all():
        fail("chunked-kl-grad", "missing/non-finite grad through checkpoint")

    run.log({k: 0 for k in MANDATORY_KEYS})
    print("[train startup_validation] PASS (gamma=1 dense check, chunked "
          "CE/KL vs full + backward, wandb keys)", flush=True)


def main():
    args = parse_args()
    import os

    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"wp2_{args.arm}{args.tag}_s{args.seq_len}_n{args.steps}",
                     config=vars(args))

    model, tokenizer = build_model(args)
    startup_validation(model, args, run)

    teacher = None
    if args.teacher_checkpoint:
        if args.arm != "distill":  # not assert: must survive python -O
            raise SystemExit("--teacher-checkpoint is distill-only; "
                             f"got --arm {args.arm}")
        teacher = build_teacher(args.teacher_checkpoint)
    lm_head_w = model.get_base_model().lm_head.weight  # frozen (LoRA is q/k/v/o)

    data = packed_stream(tokenizer, args.seq_len, source=args.data_source,
                         seed=args.data_seed)
    heldout = [next(data) for _ in range(2)]  # drift probe sequences

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr_min)

    floor = args.min_tokens_per_sec
    t_hist = []
    bench = {"fwd": [], "bwd": [], "opt": [], "step": []}
    for step in range(1, args.steps + 1):
        batch = next(data).cuda()
        if args.bench:
            if step == args.bench_warmup + 1:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.monotonic()
        if args.arm == "distill":
            # teacher under DENSE attention: the target is "what dense would
            # have output here" — CE only pressures next-token logits; this
            # pressures the whole distribution toward the dense one the
            # pipeline approximates. Default teacher = the frozen base model
            # (adapters off, same weights); --teacher-checkpoint swaps in a
            # separate frozen finetuned model instead
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(input_ids=batch).logits
            else:
                with torch.no_grad(), model.disable_adapter(), attn_impl(model, "sdpa"):
                    t_logits = model(input_ids=batch).logits
            out = model(input_ids=batch)
            loss = kl_to_teacher(out.logits, t_logits)
            if args.distill_alpha > 0:
                ce = chunked_ce(out.logits, batch)
                loss = loss + args.distill_alpha * ce
            else:
                with torch.no_grad():  # logging only
                    ce = chunked_ce(out.logits, batch)
            del t_logits
        else:
            out = model(input_ids=batch)  # no_lm_head -> logits ARE hidden states
            loss = chunked_ce_hidden(out.logits, lm_head_w, batch)
            ce = loss
        if not torch.isfinite(loss):
            print(f"[train] FATAL: non-finite loss at step {step}", flush=True)
            sys.exit(1)
        if args.bench:
            torch.cuda.synchronize()
        t_fwd = time.monotonic()
        loss.backward()
        if args.bench:
            torch.cuda.synchronize()
        t_bwd = time.monotonic()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if args.bench:
            torch.cuda.synchronize()
            if step > args.bench_warmup:
                t_end = time.monotonic()
                bench["fwd"].append(t_fwd - t0)
                bench["bwd"].append(t_bwd - t_fwd)
                bench["opt"].append(t_end - t_bwd)
                bench["step"].append(t_end - t0)
        dt = time.monotonic() - t0
        t_hist.append(dt)
        tps = args.seq_len / dt

        # ppl always from CE so it means the same thing across arms; for the
        # distill arm "loss" is the KL objective and ce_loss tracks CE
        log = {"loss": loss.item(), "ce_loss": ce.item(),
               "ppl": math.exp(min(ce.item(), 20.0)),
               "lr": sched.get_last_lr()[0], "tokens_per_sec": tps, "step": step}
        # probes run delta-pipeline math for EVERY arm and allocate inside
        # the peak-memory window (07-20 review: they set an identical
        # 20.36GB peak for all arms at 8K) — under --bench they are noise
        if (step % args.probe_every == 0 or step == args.steps) \
                and not args.bench:
            # NOTE: as of this commit anchor_grad_ratio is arm-faithful
            # (mirrors the arm's detach setting); pilot runs atyhqiir/2hepajmg
            # logged the full-graph value for every arm — not comparable.
            log["anchor_grad_ratio"] = probe_anchor_grad_ratio(
                model, heldout[0], args.gamma, args.window,
                detach_delta=model.config.detach_delta,
                delta_grad_scale=args.delta_grad_scale)  # arm-faithful
            # decomposition (delta arm only — for dense it is an expensive
            # no-op unrelated to its sdpa training graph): how much of the
            # ratio is the gamma-fold summed correction vs the anchor's
            # key-dense forward
            if args.arm in ("delta", "distill") and not model.config.detach_delta:
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

    if args.bench and bench["step"]:
        import csv
        import subprocess
        from pathlib import Path
        n = len(bench["step"])
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        # record clocks/temp AT THE END OF TIMED WORK: a thermally throttled
        # GPU (box 31, 07-20: 495/1980 MHz @ 87C) silently produces 2-3x
        # slower, ratio-distorted numbers — this column makes it visible
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            sm, sm_max, temp = [x.strip() for x in q.split(",")[:3]]
        except Exception:
            sm, sm_max, temp = "?", "?", "?"
        mean = {k: sum(v) / n for k, v in bench.items()}
        sem = {k: (sum((x - mean[k]) ** 2 for x in v) / max(n - 1, 1)) ** 0.5
               / n ** 0.5 for k, v in bench.items()}
        tok_s = args.seq_len / mean["step"]
        out = Path("results/trainbench.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        new = not out.exists()
        with out.open("a", newline="") as f:
            wtr = csv.writer(f)
            if new:
                wtr.writerow(["arm", "impl", "data_source", "seq_len",
                              "n_timed", "fwd_ms", "fwd_sem", "bwd_ms",
                              "bwd_sem", "opt_ms", "step_ms", "step_sem",
                              "tok_per_s", "peak_mem_gb", "sm_mhz",
                              "sm_max_mhz", "gpu_temp_c", "run_id"])
            impl = args.dense_impl if args.arm == "dense" else "flex_delta"
            wtr.writerow([args.arm, impl, args.data_source, args.seq_len, n]
                         + [f"{1000*mean['fwd']:.1f}", f"{1000*sem['fwd']:.1f}",
                            f"{1000*mean['bwd']:.1f}", f"{1000*sem['bwd']:.1f}",
                            f"{1000*mean['opt']:.1f}",
                            f"{1000*mean['step']:.1f}", f"{1000*sem['step']:.1f}",
                            f"{tok_s:.0f}", f"{peak_gb:.2f}",
                            sm, sm_max, temp, run.id])
        for k in ("fwd", "bwd", "opt", "step"):
            run.summary[f"bench_{k}_ms"] = 1000 * mean[k]
        run.summary["bench_tok_per_s"] = tok_s
        run.summary["bench_peak_mem_gb"] = peak_gb
        print(f"[bench] {args.arm} @{args.seq_len}: "
              f"fwd {1000*mean['fwd']:.0f}ms bwd {1000*mean['bwd']:.0f}ms "
              f"step {1000*mean['step']:.0f}ms ({tok_s:.0f} tok/s, "
              f"peak {peak_gb:.1f}GB, n={n})", flush=True)

    model.save_pretrained(args.save_dir)
    if not args.no_artifact:
        # boxes self-terminate on completion — the adapter must outlive the
        # disk. LoRA r16 on q/k/v/o is ~50MB; wandb artifact is the durable
        # home.
        meta = {"steps": args.steps, "seq_len": args.seq_len,
                "gamma": args.gamma, "arm": args.arm, "tag": args.tag,
                "delta_grad_scale": args.delta_grad_scale,
                "data_seed": args.data_seed}
        if args.arm == "distill":  # distill-only fields; a CE arm claiming a
            meta["distill_alpha"] = args.distill_alpha  # teacher would mislabel it
            meta["teacher_checkpoint"] = args.teacher_checkpoint
            if args.teacher_checkpoint:
                # provenance: fetch_adapters records the RESOLVED artifact
                # version+digest here; ':latest' alone is a floating alias and
                # would make the run irreproducible if the teacher is re-uploaded
                from pathlib import Path
                pin = Path(args.teacher_checkpoint) / "WANDB_ARTIFACT"
                meta["teacher_artifact"] = (
                    pin.read_text().strip() if pin.exists() else "UNPINNED")
        artifact = wandb.Artifact(f"wp2_adapter_{args.arm}{args.tag}",
                                  type="lora-adapter", metadata=meta)
        artifact.add_dir(args.save_dir)
        run.log_artifact(artifact)
    run.summary["final_loss"] = loss.item()
    run.finish()  # blocks until artifact upload completes
    print(f"[train] DONE: {args.steps} steps, final loss {loss.item():.4f}, "
          f"adapter saved to {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
