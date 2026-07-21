"""Track A trainer: MTP draft module on a frozen dense Llama trunk.

Fully parallel teacher forcing — no sequential chaining at train time:
module position t gets (h_t, Emb(x_{t+1})) and is trained on CE against
x_{t+2} across the whole sequence at once. Trunk forward is no-grad; only
the ~220M-param module trains.

    python -m delta_attention.train.train_mtp --steps 2000 --seq-len 8192 \
        --module-attn delta --save-path checkpoints/mtp_delta.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

from delta_attention.mtp import MTPModule
from delta_attention.train.train_delta import chunked_ce_hidden, packed_stream


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--module-attn", choices=["dense", "delta"], required=True)
    p.add_argument("--gamma", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--data-source", choices=["pg19", "arxiv"], default="pg19")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--model", type=str, default="")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--save-path", type=str, required=True)
    return p.parse_args()


def build_trunk(model_str=""):
    """Frozen DENSE trunk (Phase A holds the trunk constant; only the
    module's attention varies)."""
    from delta_attention.config import Config
    from delta_attention.sample import init_model

    cfg = Config()
    if model_str:
        cfg.model_str = model_str
    cfg.attn_implementation = "flash_attention_2"
    cfg.mode = "none"
    cfg.attn_implementation_original = cfg.attn_implementation
    trunk, tokenizer = init_model(cfg)
    trunk.config.use_cache = False
    for p_ in trunk.parameters():
        p_.requires_grad_(False)
    setattr(trunk, "no_lm_head", True)  # out.logits -> final hidden states
    return trunk.eval().cuda(), tokenizer


def main():
    args = parse_args()
    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"mtp_{args.module_attn}{args.tag}", config=vars(args))
    trunk, tokenizer = build_trunk(args.model)
    module = MTPModule(trunk, module_attn=args.module_attn,
                       gamma=args.gamma, window=args.window)
    module = module.to(torch.bfloat16).cuda().train()
    n_train = sum(p_.numel() for p_ in module.parameters())
    print(f"[mtp] module params: {n_train/1e6:.0f}M ({args.module_attn})",
          flush=True)

    lm_w = trunk.lm_head.weight  # frozen shared head
    embed = trunk.model.embed_tokens
    rotary = trunk.model.rotary_emb

    # pre-spend validation (07-21 review: train_delta's startup_validation
    # has no counterpart here): one tiny forward+backward through the
    # module's actual attention branch must produce finite loss and grads
    # 4096 > sink+window (3072): the delta branch's sparse mask is real
    # here (07-21 review round 2: a 192-length validation had every key
    # inside the sink, making the "sparse" path identical to dense and the
    # validation vacuous for --module-attn delta). 4096 is a 64-multiple.
    _b = torch.randint(0, trunk.config.vocab_size, (1, 4160), device="cuda")
    with torch.no_grad():
        _h = trunk(input_ids=_b).logits
    _mh = module.forward_parallel(_h[:, :4096], embed(_b[:, 1:4097]), rotary)
    _l = chunked_ce_hidden(_mh, lm_w, _b[:, 1:4097])
    _l.backward()
    assert torch.isfinite(_l), "startup validation: non-finite loss"
    for _n, _p in module.named_parameters():
        assert _p.grad is not None and torch.isfinite(_p.grad).all(), \
            f"startup validation: bad grad for {_n}"
    module.zero_grad(set_to_none=True)
    print("[mtp] startup validation PASS", flush=True)

    data = packed_stream(tokenizer, args.seq_len, source=args.data_source,
                         seed=args.data_seed)
    opt = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr_min)

    for step in range(1, args.steps + 1):
        batch = next(data).cuda()  # [1, L]
        t0 = time.monotonic()
        with torch.no_grad():
            h = trunk(input_ids=batch).logits  # [1, L, d] hidden states
        # module position t: inputs (h_t, Emb(x_{t+1})), t = 0..L-2.
        # Length trimmed to a 64-multiple: the flex delta kernel asserts
        # s % 64 == 0, and L-1 = 8191 is not (07-21 review — this would
        # have crashed only AFTER the paid dense run). Same trim for both
        # variants keeps the comparison symmetric.
        m_len = ((batch.size(1) - 1) // 64) * 64
        hiddens = h[:, :m_len]
        tok_embs = embed(batch[:, 1:m_len + 1])
        mh = module.forward_parallel(hiddens, tok_embs, rotary)
        # chunked_ce_hidden pairs hd[:, :-1] with tg[:, 1:]:
        # hd pos t (=module pos t, predicts x_{t+2}) vs tg[t+1] = x_{t+2}  ✓
        loss = chunked_ce_hidden(mh, lm_w, batch[:, 1:m_len + 1])
        if not torch.isfinite(loss):
            print(f"[mtp] FATAL: non-finite loss at step {step}", flush=True)
            sys.exit(1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        dt = time.monotonic() - t0
        run.log({"loss": loss.item(), "ppl": math.exp(min(loss.item(), 20.0)),
                 "lr": sched.get_last_lr()[0],
                 "tokens_per_sec": args.seq_len / dt, "step": step})
        if step % 100 == 0 or step == 1:
            print(f"[mtp] step {step}: loss {loss.item():.4f} "
                  f"({args.seq_len/dt:.0f} tok/s)", flush=True)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save({"state_dict": module.state_dict(),
                "module_attn": args.module_attn, "gamma": args.gamma,
                "window": args.window, "steps": args.steps,
                "seq_len": args.seq_len, "run_id": run.id}, args.save_path)
    art = wandb.Artifact(f"mtp_module_{args.module_attn}{args.tag}",
                         type="mtp-module",
                         metadata={"steps": args.steps, "attn": args.module_attn})
    art.add_file(args.save_path)
    run.log_artifact(art)
    run.summary["final_loss"] = loss.item()
    run.finish()
    print(f"[mtp] DONE: final loss {loss.item():.4f} -> {args.save_path}",
          flush=True)


if __name__ == "__main__":
    main()
