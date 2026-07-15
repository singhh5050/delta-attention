"""True speculative decoding: draft (sparse/delta decode) + verify (dense).

Jeff's hypothesis, operationalized: the draft proposes a block of K tokens
using local-context attention; ONE dense rectangle forward verifies the whole
block; tokens are accepted sequentially until the first mismatch (then the
dense model's token is substituted — the "target takes over"). Greedy
everywhere, so the output is EXACTLY what the dense-decode arm would have
produced (base_delta: delta prefill + dense decode) — quality is dense by
construction, the results are acceptance/cost numbers.

Cache discipline (single shared KV cache — same weights for draft & target):
  - L_dense = prefix length whose KV is dense-grade (prefill + verify passes)
  - draft forwards append draft-grade KV for pending+proposed tokens
  - each verify CROPS the cache back to L_dense and re-forwards
    pending+proposals as one dense rectangle block, leaving dense-grade KV
    for everything it accepts (cropped again past the first rejection)

Delta-draft semantics: _dec_state is reset per draft phase, so each block is
one dense-row anchor + (K-1) cached-delta reuses — Jeff's gamma_dec concept
with block size as gamma.

    python eval/specdec_eval.py --suite govreport --n-samples 20 \
        --drafts sparse,delta --blocks 2,4,8
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

try:  # keep the pure acceptance logic importable on torch-less boxes
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.longbench_eval import (  # noqa: E402
    GOVREPORT_MAX_NEW, GOVREPORT_TEMPLATE, MAX_PROMPT_TOKENS,
    V1_TASKS, chat_wrap, load_v1, truncate_middle,
)

DRAFT_MODES = ("sparse", "delta")
# draft weights: base model or a trained adapter (merged at init)
DRAFT_WEIGHTS = {"base": "", "ce32k": "checkpoints/pilot_delta_32k",
                 "dft": "checkpoints/pilot_distill_dft"}


def accept_block(proposals, dense_choices, bonus):
    """proposals[i] is accepted iff proposals[j] == dense_choices[j] for all
    j <= i. dense_choices[i] = dense greedy token at proposals[i]'s slot.
    bonus = dense greedy token after the LAST proposal (used only if the
    whole block is accepted). Returns (n_accepted, next_token, full_block)."""
    n = 0
    for p, d in zip(proposals, dense_choices):
        if p != d:
            return n, d, False  # dense takes over at first mismatch
        n += 1
    return n, bonus, True


def build(draft_weights_key, gamma_prefill, window):
    from delta_attention.config import Config
    from delta_attention.sample import init_model

    cfg = Config()
    cfg.attn_implementation = "window"
    cfg.mode = "delta"
    cfg.delta_lambda = gamma_prefill
    cfg.sliding_window = window
    cfg.attn_implementation_original = cfg.attn_implementation
    ckpt = DRAFT_WEIGHTS[draft_weights_key]
    if ckpt:
        assert Path(ckpt).exists(), f"adapter missing: {ckpt}"
        cfg.checkpoint = ckpt
    model, tokenizer = init_model(cfg)
    model.config.log_drift = False
    model.eval().cuda()
    return model, tokenizer


def _reset_dec_state(model):
    from delta_attention.llama import LlamaAttention

    for m in model.modules():
        if isinstance(m, LlamaAttention):
            m._dec_state = None
            m._dec_drift_points = None


def _forward_tokens(model, ids_1d, cache):
    """Feed ids (list[int]) as one block; returns (logits_all_positions, cache)."""
    inp = torch.tensor([ids_1d], device="cuda")
    with torch.no_grad():
        out = model(input_ids=inp, past_key_values=cache, use_cache=True)
        hidden = out.logits  # no_lm_head -> hidden states
        logits = model.lm_head(hidden).float()[0]  # [len, vocab]
    return logits, out.past_key_values


def spec_generate(model, tokenizer, prompt, draft_mode, block, max_new):
    """Returns (token_ids, stats). Greedy; exact dense-arm output."""
    setattr(model, "no_lm_head", True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])

    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False)["input_ids"][0].tolist()

    # prefill under the pipeline (mirrors _sample), then rectangle decode
    _reset_dec_state(model)
    model.config._attn_implementation = model.config.attn_implementation_original
    t0 = time.monotonic()
    with torch.no_grad():
        out = model(torch.tensor([ids], device="cuda"), use_cache=True)
        first_logits = model.lm_head(out.logits[:, -1, :]).float()[0]
    model.config._attn_implementation = "sdpa_rectangle"
    cache = out.past_key_values
    t_prefill = time.monotonic() - t0

    L_dense = len(ids)                      # dense-grade cache length
    pending = [int(first_logits.argmax())]  # accepted, KV not yet in cache
    generated = [pending[0]]
    stats = {"proposed": 0, "accepted": 0, "verify_calls": 0, "draft_calls": 0,
             "full_blocks": 0, "blocks": 0, "first_reject_pos": [],
             "t_prefill": t_prefill, "t_draft": 0.0, "t_verify": 0.0}

    while len(generated) < max_new and generated[-1] not in eos:
        # ---- draft phase: consume pending, then propose block-1 more ----
        model.config.decode_mode = draft_mode
        _reset_dec_state(model)  # each block: anchor + (K-1) delta reuses
        t0 = time.monotonic()
        proposals = []
        cur = list(pending)
        for _ in range(block):
            logits, cache = _forward_tokens(model, [cur[-1]] if len(cur) == 1
                                            else cur, cache)
            stats["draft_calls"] += 1
            nxt = int(logits[-1].argmax())
            proposals.append(nxt)
            cur = [nxt]
        # cache now holds draft-grade KV for pending + proposals[:-1]
        stats["t_draft"] += time.monotonic() - t0

        # ---- verify: crop to dense-grade, one rectangle block ----
        model.config.decode_mode = "dense"
        cache.crop(L_dense)
        t0 = time.monotonic()
        blk = pending + proposals
        logits, cache = _forward_tokens(model, blk, cache)
        stats["verify_calls"] += 1
        stats["t_verify"] += time.monotonic() - t0
        # dense choice at proposals[i]'s slot = argmax after its predecessor
        m = len(pending)
        dense_choices = [int(logits[m - 1 + i].argmax()) for i in range(len(proposals))]
        bonus = int(logits[-1].argmax())

        n_acc, nxt, full = accept_block(proposals, dense_choices, bonus)
        stats["proposed"] += len(proposals)
        stats["accepted"] += n_acc
        stats["blocks"] += 1
        stats["full_blocks"] += int(full)
        if not full:
            stats["first_reject_pos"].append(n_acc)
        # dense-grade KV: pending + accepted proposals stay; rest cropped
        keep = L_dense + m + n_acc
        cache.crop(keep)
        L_dense = keep

        emitted = proposals[:n_acc] + [nxt]
        for t in emitted:
            generated.append(t)
            if len(generated) >= max_new or t in eos:
                break
        pending = [generated[-1]] if generated[-1] not in eos else []
        if not pending:
            break

    setattr(model, "no_lm_head", True)
    return generated, stats


def dense_reference(model, tokenizer, prompt, max_new):
    """Plain dense-decode generation (the exactness target), via generate()."""
    model.config.decode_mode = "dense"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        t0 = time.monotonic()
        out = model.generate(**{k: v.cuda() for k, v in inputs.items()},
                             max_new_tokens=max_new, do_sample=False)
        dt = time.monotonic() - t0
    return out[0, inputs["input_ids"].size(1):].tolist(), dt


def load_prompts(suite, tokenizer, n):
    if suite == "govreport":
        rows = load_v1("gov_report")[:n]
        prompts = [truncate_middle(GOVREPORT_TEMPLATE.replace("{context}", r["context"]),
                                   tokenizer, MAX_PROMPT_TOKENS) for r in rows]
        return prompts, GOVREPORT_MAX_NEW
    if suite == "qa":
        prompts = []
        per = max(n // len(V1_TASKS), 1)
        for task, (template, _mx) in V1_TASKS.items():
            for ex in load_v1(task)[:per]:
                p = template.format(context=ex["context"], input=ex["input"])
                prompts.append(truncate_middle(p, tokenizer, MAX_PROMPT_TOKENS))
        return prompts, 64
    raise SystemExit(f"unknown suite {suite}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["govreport", "qa"], required=True)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--drafts", type=str, default="sparse,delta")
    p.add_argument("--blocks", type=str, default="2,4,8")
    p.add_argument("--weights", type=str, default="base",
                   help=f"comma list from {list(DRAFT_WEIGHTS)}")
    p.add_argument("--gamma", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--exact-check-n", type=int, default=3,
                   help="verify exact dense parity on the first N prompts "
                        "per config (dense reference is a second full "
                        "generation — expensive, so not on every sample)")
    p.add_argument("--min-parity-prefix", type=int, default=0,
                   help="smoke gate: hard-fail if any parity check diverges "
                        "from the sequential dense reference BEFORE this many "
                        "tokens. Bitwise parity is unattainable (verify writes "
                        "KV with q_len=k kernels, sequential with q_len=1 — "
                        "bf16 reduction order differs and a near-tie argmax "
                        "eventually flips; observed at token ~75, never early). "
                        "Systematic harness bugs diverge at tokens 0-10, so an "
                        "early divergence = bug. 0 disables the gate")
    p.add_argument("--out", type=str, default="results/specdec.csv")
    args = p.parse_args()

    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"specdec_{args.suite}", config=vars(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists()
    fh = out_path.open("a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["suite", "weights", "draft", "block", "n", "acceptance",
                    "acc_per_verify", "full_block_rate", "tok_per_s_spec",
                    "tok_per_s_dense", "parity_prefix_min", "run_id"])

    drafts = [d.strip() for d in args.drafts.split(",")]
    blocks = [int(b) for b in args.blocks.split(",")]
    weights = [x.strip() for x in args.weights.split(",")]
    bad = [d for d in drafts if d not in DRAFT_MODES] + \
          [x for x in weights if x not in DRAFT_WEIGHTS]
    if bad:
        raise SystemExit(f"unknown drafts/weights {bad}")

    for wkey in weights:
        model, tokenizer = build(wkey, args.gamma, args.window)
        prompts, max_new = load_prompts(args.suite, tokenizer, args.n_samples)
        prompts = [chat_wrap(pr, tokenizer) for pr in prompts]
        # dense references are draft/block-independent: compute ONCE per
        # weights config (was recomputed per config — 6x waste on the grid)
        refs = {}
        for i in range(min(args.exact_check_n, len(prompts))):
            refs[i] = dense_reference(model, tokenizer, prompts[i], max_new)
        for draft in drafts:
            for block in blocks:
                agg = {"proposed": 0, "accepted": 0, "verify_calls": 0,
                       "full_blocks": 0, "blocks": 0, "tokens": 0,
                       # timing comparison uses ONLY the checked prompts and
                       # includes prefill on BOTH sides (dense_reference's
                       # generate() includes its prefill; spec adds t_prefill)
                       "tok_chk": 0, "t_spec_chk": 0.0,
                       "tok_ref": 0, "t_dense": 0.0, "exact": []}
                for i, prompt in enumerate(prompts):
                    toks, st = spec_generate(model, tokenizer, prompt, draft,
                                             block, max_new)
                    for k in ("proposed", "accepted", "verify_calls",
                              "full_blocks", "blocks"):
                        agg[k] += st[k]
                    agg["tokens"] += len(toks)
                    if i in refs:
                        ref, dt = refs[i]
                        agg["tok_chk"] += len(toks)
                        agg["t_spec_chk"] += (st["t_prefill"] + st["t_draft"]
                                              + st["t_verify"])
                        agg["tok_ref"] += len(ref)
                        agg["t_dense"] += dt
                        n_cmp = min(len(toks), len(ref))
                        div = next((j for j in range(n_cmp)
                                    if toks[j] != ref[j]), None)
                        if div is None and len(toks) < len(ref):
                            div = len(toks)  # early truncation = divergence
                        agg["exact"].append(n_cmp if div is None else div)
                    print(f"[specdec] {wkey}/{draft}/b{block} #{i}: "
                          f"acc {st['accepted']}/{st['proposed']}", flush=True)
                acc = agg["accepted"] / max(agg["proposed"], 1)
                apv = agg["accepted"] / max(agg["verify_calls"], 1)
                fbr = agg["full_blocks"] / max(agg["blocks"], 1)
                tps_spec = agg["tok_chk"] / max(agg["t_spec_chk"], 1e-9)
                tps_dense = agg["tok_ref"] / max(agg["t_dense"], 1e-9)
                parity = min(agg["exact"]) if agg["exact"] else None
                if (args.min_parity_prefix and parity is not None
                        and parity < args.min_parity_prefix):
                    raise SystemExit(
                        f"PARITY FAILURE: {wkey}/{draft}/b{block} diverged from "
                        f"dense greedy at token {parity} (< "
                        f"{args.min_parity_prefix}) — early divergence means a "
                        "harness bug, not kernel numerics; aborting")
                w.writerow([args.suite, wkey, draft, block, len(prompts),
                            f"{acc:.4f}", f"{apv:.3f}", f"{fbr:.4f}",
                            f"{tps_spec:.2f}", f"{tps_dense:.2f}",
                            parity, run.id])
                fh.flush()
                run.log({f"specdec/{wkey}/{draft}/b{block}/acceptance": acc,
                         f"specdec/{wkey}/{draft}/b{block}/acc_per_verify": apv,
                         f"specdec/{wkey}/{draft}/b{block}/full_block_rate": fbr})
                run.summary[f"acc_{wkey}_{draft}_b{block}"] = acc
                print(f"[specdec] {wkey}/{draft}/b{block}: acceptance {acc:.3f} "
                      f"acc/verify {apv:.2f} full-block {fbr:.3f} "
                      f"parity_prefix_min={parity}", flush=True)
        model._sample = None
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    fh.close()
    run.finish()
    print("[specdec] DONE", flush=True)


if __name__ == "__main__":
    main()
