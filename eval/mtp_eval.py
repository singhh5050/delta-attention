"""Track A eval: true speculative decoding with the MTP module as drafter.

Trunk = frozen DENSE Llama (prefill, hidden extraction, verify) — output is
exactly dense-greedy by construction (same accept_block as specdec_eval).
Module proposes K tokens per trunk verify by chaining (depth-1-trained; at
K>1 the module feeds its own hidden back as the trunk-hidden stand-in).

Position indexing (y_i = token at trunk position i, h_i = trunk hidden AT
y_i; module position t inputs (h_t, Emb(y_{t+1})) and predicts y_{t+2}):
- prompt of length L: module cache prefilled over positions 0..L-2
- to propose the token after pending y_N: module position N-1 gets
  (h_{N-1}, Emb(y_N))
- after a verify that emits `consumed` tokens, the K speculative module
  cache entries are cropped and `consumed` CONFIRMED entries are appended
  from TRUE trunk hiddens (the verify rectangle computes them for free).

Acceptance here has NO structural free position (the module is not the
trunk), so headline acceptance IS the genuine number — but per-position
curves are logged anyway; the K>1 decay measures chaining degradation.

    python eval/mtp_eval.py --suite govreport --n-samples 10 \
        --modules checkpoints/mtp_dense.pt,checkpoints/mtp_delta.pt --blocks 1,2,4
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from delta_attention.mtp import MTPModule  # noqa: E402
from eval.specdec_eval import (  # noqa: E402
    TIE_EPS, accept_block, classify_parity, load_prompts, positional_stats,
)


def build_trunk(model_str=""):
    from delta_attention.config import Config
    from delta_attention.sample import init_model

    cfg = Config()
    if model_str:
        cfg.model_str = model_str
    cfg.attn_implementation = "flash_attention_2"
    cfg.mode = "none"
    cfg.attn_implementation_original = cfg.attn_implementation
    trunk, tokenizer = init_model(cfg)
    trunk.config.use_cache = True
    setattr(trunk, "no_lm_head", True)
    return trunk.eval().cuda(), tokenizer


def fwd(trunk, ids_1d, cache):
    """Forward ids; returns (logits[len,vocab] fp32, hidden[len,d], cache)."""
    inp = torch.tensor([ids_1d], device="cuda")
    with torch.no_grad():
        out = trunk(input_ids=inp, past_key_values=cache, use_cache=True)
        hidden = out.logits[0]  # no_lm_head
        logits = trunk.lm_head(hidden).float()
    return logits, hidden, out.past_key_values


def mtp_generate(trunk, module, tokenizer, prompt, K, max_new):
    embed_w = trunk.model.embed_tokens.weight
    rotary = trunk.model.rotary_emb
    lm_w = trunk.lm_head.weight
    eos = trunk.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False)["input_ids"][0].tolist()
    L = len(ids)

    with torch.no_grad():
        out = trunk(input_ids=torch.tensor([ids], device="cuda"),
                    use_cache=True)
        hidden_prompt = out.logits[0]  # [L, d]
        first_logits = trunk.lm_head(hidden_prompt[-1:]).float()[0]
    cache = out.past_key_values
    module.prefill_cache(hidden_prompt[None, :-1, :],
                         trunk.model.embed_tokens(
                             torch.tensor([ids[1:]], device="cuda")),
                         rotary)

    L_dense = L
    pending = [int(first_logits.argmax())]
    h_pend = hidden_prompt[-1]  # h_{L-1}
    generated = [pending[0]]
    stats = {"proposed": 0, "accepted": 0, "blocks": 0, "full_blocks": 0,
             "nacc": [], "t_draft": 0.0, "t_verify": 0.0}

    while len(generated) < max_new and generated[-1] not in eos:
        base = module.cache_len()
        t0 = time.monotonic()
        proposals = []
        cur_h, cur_tok = h_pend, pending[0]
        with torch.no_grad():
            for j in range(K):
                mh = module.decode_step(cur_h, embed_w[cur_tok], rotary,
                                        pos=base + j)
                # bf16 matmul; fp32 cast of lm_w here would copy ~2GB per
                # drafted token (07-21 review). Draft argmax needs no fp32
                # exactness — the verify decides acceptance either way.
                proposals.append(int((mh @ lm_w.T).float().argmax()))
                cur_h, cur_tok = mh, proposals[-1]
        torch.cuda.synchronize()
        stats["t_draft"] += time.monotonic() - t0

        t0 = time.monotonic()
        blk = pending + proposals
        logits, hidden_blk, cache = fwd(trunk, blk, cache)
        torch.cuda.synchronize()
        stats["t_verify"] += time.monotonic() - t0
        m = len(pending)
        dense_choices = [int(logits[m - 1 + i].argmax())
                         for i in range(len(proposals))]
        bonus = int(logits[-1].argmax())
        n_acc, nxt, full = accept_block(proposals, dense_choices, bonus)
        keep = L_dense + m + n_acc
        cache.crop(keep)
        L_dense = keep

        emitted = proposals[:n_acc] + [nxt]
        consumed = 0
        for t in emitted:
            generated.append(t)
            consumed += 1
            if len(generated) >= max_new or t in eos:
                break
        # module cache: drop speculative entries, append CONFIRMED ones
        module.crop(base)
        block_toks = pending + emitted
        with torch.no_grad():
            for i in range(consumed):
                h_i = h_pend if i == 0 else hidden_blk[i - 1]
                # module pos t pairs h_t with Emb(y_{t+1}) = block_toks[i]
                # (07-21 review: [i+1] silently corrupted every confirmed
                # cache entry with next-NEXT-token embeddings)
                module.decode_step(h_i, embed_w[block_toks[i]], rotary,
                                   pos=base + i)
        # count ONLY blocks fully inside the emitted sequence — the same
        # deliberate protocol as specdec_eval (07-16 review): a final block
        # cut by eos/max_new contains proposals sequential dense decoding
        # would never have needed
        if consumed == len(emitted):
            stats["proposed"] += len(proposals)
            stats["accepted"] += n_acc
            stats["blocks"] += 1
            stats["full_blocks"] += int(full)
            stats["nacc"].append(n_acc)
        h_pend = hidden_blk[consumed - 1]
        pending = [generated[-1]] if generated[-1] not in eos else []
        if not pending:
            break
    return generated, stats


def dense_reference(trunk, tokenizer, prompt, max_new):
    trunk.config.use_cache = True
    # generate() needs REAL logits. llama.py gates on hasattr(no_lm_head)
    # — the VALUE is ignored (07-21 review: setattr(False) left generate
    # emitting hidden-state argmaxes) — so the only correct disable is
    # delattr
    if hasattr(trunk, "no_lm_head"):
        delattr(trunk, "no_lm_head")
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = trunk.generate(**{k: v.cuda() for k, v in inputs.items()},
                             max_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()
    setattr(trunk, "no_lm_head", True)
    return out[0, inputs["input_ids"].size(1):].tolist()


def margin_at(trunk, tokenizer, prompt, ref, div, spec_tok):
    """Tie detector — trunk is plain dense here, so one no-cache forward IS
    the exact reference semantics (unlike specdec's pipeline case)."""
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False)["input_ids"][0].tolist()
    with torch.no_grad():
        out = trunk(input_ids=torch.tensor([ids + ref[:div]], device="cuda"),
                    use_cache=False)
        logits = trunk.lm_head(out.logits[0][-1:]).float()[0]
    return float(logits.max() - logits[spec_tok])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["govreport", "qa", "ruler"],
                   required=True)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--modules", type=str, required=True,
                   help="comma list of module checkpoint .pt paths")
    p.add_argument("--blocks", type=str, default="1,2,4")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--exact-check-n", type=int, default=3)
    p.add_argument("--min-parity-prefix", type=int, default=0)
    p.add_argument("--min-smoke-acceptance", type=float, default=0.0,
                   help="hard-fail if K=1 acceptance falls below this "
                        "(smoke gate: a position-indexing bug reads as "
                        "near-zero acceptance, not as a crash)")
    p.add_argument("--out", type=str, default="results/mtp.csv")
    args = p.parse_args()

    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"mtp_eval_{args.suite}", config=vars(args))
    from eval.specdec_eval import open_csv
    fh, w = open_csv(Path(args.out),
                     ["suite", "module", "module_attn", "K", "n", "acceptance",
                      "pos_acc", "acc_per_verify", "full_block_rate",
                      "parity_prefix_min", "run_id"])

    trunk, tokenizer = build_trunk(args.model)
    prompts, budgets = load_prompts(args.suite, tokenizer, args.n_samples)
    refs = {i: dense_reference(trunk, tokenizer, prompts[i], budgets[i])
            for i in range(min(args.exact_check_n, len(prompts)))}

    for mod_path in [m.strip() for m in args.modules.split(",")]:
        ckpt = torch.load(mod_path, map_location="cpu", weights_only=False)
        module = MTPModule(trunk, module_attn=ckpt["module_attn"],
                           gamma=ckpt["gamma"], window=ckpt["window"])
        module.load_state_dict(ckpt["state_dict"])
        module = module.to(torch.bfloat16).cuda().eval()
        mname = Path(mod_path).stem
        for K in [int(b) for b in args.blocks.split(",")]:
            agg = {"proposed": 0, "accepted": 0, "blocks": 0,
                   "full_blocks": 0, "nacc": [], "exact": []}
            for i, prompt in enumerate(prompts):
                toks, st = mtp_generate(trunk, module, tokenizer, prompt, K,
                                        budgets[i])
                for k_ in ("proposed", "accepted", "blocks", "full_blocks"):
                    agg[k_] += st[k_]
                agg["nacc"].extend(st["nacc"])
                if i in refs:
                    ref = refs[i]
                    n_cmp = min(len(toks), len(ref))
                    div = next((j for j in range(n_cmp)
                                if toks[j] != ref[j]), None)
                    eos_ids = trunk.generation_config.eos_token_id
                    eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple))
                                  else [eos_ids])
                    if div is not None:
                        parity_i = div
                        if args.min_parity_prefix and div < args.min_parity_prefix:
                            mg = margin_at(trunk, tokenizer, prompt, ref, div,
                                           toks[div])
                            print(f"[mtp] early divergence @{div}: margin "
                                  f"{mg:.3f}", flush=True)
                            if mg <= TIE_EPS:
                                parity_i = ("tie", div, mg)
                    elif len(toks) != len(ref):
                        parity_i = -1
                    elif toks and toks[-1] in eos_ids:
                        parity_i = 10**9
                    elif n_cmp >= max(args.min_parity_prefix, 1):
                        parity_i = 10**9
                    else:
                        parity_i = n_cmp
                    agg["exact"].append(parity_i)
                print(f"[mtp] {mname}/K{K} #{i}: acc "
                      f"{st['accepted']}/{st['proposed']}", flush=True)
            acc = agg["accepted"] / max(agg["proposed"], 1)
            apv = agg["accepted"] / max(agg["blocks"], 1)
            fbr = agg["full_blocks"] / max(agg["blocks"], 1)
            pos_acc, _ = positional_stats(agg["nacc"], K)
            parity, pfail = classify_parity(agg["exact"],
                                            args.min_parity_prefix)
            if pfail:
                raise SystemExit(f"PARITY FAILURE: {mname}/K{K} {pfail}")
            if (args.min_smoke_acceptance and K == 1
                    and acc < args.min_smoke_acceptance):
                raise SystemExit(
                    f"ACCEPTANCE GATE: {mname}/K1 acceptance {acc:.3f} < "
                    f"{args.min_smoke_acceptance} — position-indexing bug "
                    "or untrained module")
            w.writerow([args.suite, mname, ckpt["module_attn"], K,
                        len(prompts), f"{acc:.4f}",
                        ";".join(f"{x:.4f}" for x in pos_acc),
                        f"{apv:.3f}", f"{fbr:.4f}", parity, run.id])
            fh.flush()
            run.summary[f"acc_{mname}_K{K}"] = acc
            print(f"[mtp] {mname}/K{K}: acceptance {acc:.3f} "
                  f"pos {';'.join(f'{x:.2f}' for x in pos_acc)} "
                  f"parity={parity}", flush=True)
        del module
        gc.collect()
        torch.cuda.empty_cache()
    fh.close()
    run.finish()
    print("[mtp] DONE", flush=True)


if __name__ == "__main__":
    main()
