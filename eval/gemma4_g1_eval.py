"""Gemma 4 G1: delta-read the native MTP drafter (docs/gemma4_mtp_plan.md).

STANDALONE (runs in .venv-g4, transformers MAIN, no delta_attention
imports). G0 certified the native pipeline; G1 owns the draft-verify loop
so the drafter's cross-read over the trunk KV becomes an intervention
point. THE DRAFTER IS NEVER REIMPLEMENTED OR MODIFIED — the drafting
recipe below is transcribed from transformers' Gemma4 candidate generator
(generation/candidate_generator.py, get_candidates):

    - drafter is called ONE token at a time, use_cache=False
    - inputs_embeds = cat([target_embed(last_token_id),
                           last_hidden_state_of_last_validated], dim=-1)
    - position_ids CONSTANT during a round (= len(validated) - 1)
    - within a round it chains on its OWN argmax + post_projection hidden
    - it cross-attends shared_kv_states (the TRUNK's full-length KV per
      layer type) cropped to the validated prefix

The arms differ ONLY in the shared_kv_states dict handed to the drafter:
    full     — cropped, untouched (the certification arm: gated against
               PLAIN trunk greedy computed with identical forward shapes
               — the spec-decode exactness invariant; a short-context
               native-assisted cross-check guards shared-bug blindness)
    sparse   — the full_attention entry subsampled to sink+window
               (StreamingLLM-style read; rope is baked into the trunk's
               cached keys, so gathering preserves positions)
    deltaN   — sparse, but every N-th drafter call in a round reads the
               full entry (anchor-refresh cadence)
Acceptance is verified by the TRUNK (greedy, strict prefix + bonus), so
"accepted tokens per round" is exact and comparable across arms. Per-call
drafter latency is logged per arm — that is the G2 read-cost data.

Trunk-side notes: prefill uses logits_to_keep=1 (full-vocab prefill
logits were the >=16K OOM in G0), the trunk runs on a DynamicCache
(upstream hybrid-cache bug with multi-token verify, §U), and every
verify forward passes return_shared_kv_states=True.

    python eval/gemma4_g1_eval.py --n 2 --tiers 4096 --max-new 64 \
        --arms full --parity-check --out results/g1_smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch

TARGET = "google/gemma-4-31b-it"
ASSISTANT = "google/gemma-4-31b-it-assistant"
SINK, WINDOW = 1024, 2048  # project-standard delta read geometry
SLIDING_CAP = 1025  # native hybrid-cache sliding-entry length (window+1):
# with an uncapped/offloaded trunk cache the sliding shared-KV entry comes
# back full-length, which is NOT what the drafter's flip-mask math was
# built for — cap it to native semantics ourselves

INSTR = ("Continue this story in the same style. Write a long, natural "
         "continuation.\n\n")

HEADER = ["tier", "idx", "arm", "prompt_toks", "rounds", "drafted",
          "accepted", "acc_per_round", "acc_rate", "bonus", "total_new",
          "draft_call_ms", "match_vs_full", "run_id"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--tiers", type=str, default="4096,16384,32768")
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--k", type=int, default=5, help="drafts per round")
    p.add_argument("--arms", type=str, default="full,sparse,delta4")
    p.add_argument("--parity-check", action="store_true",
                   help="also run native generate() and require the full "
                        "arm's tokens to match it exactly (smoke gate)")
    p.add_argument("--target", type=str, default=TARGET)
    p.add_argument("--assistant", type=str, default=ASSISTANT)
    p.add_argument("--offload", action="store_true",
                   help="CPU-offloaded uncapped trunk cache: the hybrid "
                        "cache kept >=16K from fitting anyway (76GB "
                        "allocated at the 16K prefill, box 44), and "
                        "offloading fits any tier at a decode-speed cost "
                        "that cannot affect acceptance")
    p.add_argument("--out", type=str, default="results/g1_tiers.csv")
    return p.parse_args()


def load_contexts(tokenizer, n, max_ctx_tokens):
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    out = []
    for doc in ds:
        ids = tokenizer.encode(doc["text"], add_special_tokens=False)
        if len(ids) < max_ctx_tokens:
            continue  # tier label must equal actual prompt length
            # (review wf_4081ce31: >=tier/2 let ~33K docs into the 64K tier)
        out.append(tokenizer.decode(ids[:max_ctx_tokens],
                                    skip_special_tokens=True))
        if len(out) == n:
            return out
    raise SystemExit(f"only {len(out)} PG19 docs long enough for tier "
                     f"{max_ctx_tokens}")


def cap_sliding(kv):
    """Restore native sliding-entry semantics (last window+1 rows) when the
    trunk cache is uncapped/offloaded; no-op when already capped."""
    k, v = kv["sliding_attention"]
    if k.shape[2] <= SLIDING_CAP:
        return kv
    out = dict(kv)
    out["sliding_attention"] = (k[:, :, -SLIDING_CAP:], v[:, :, -SLIDING_CAP:])
    return out


def arm_view(kv, arm, call_idx):
    """The intervention: how THIS drafter call sees the trunk KV (already
    exactly the validated prefix). call_idx is GLOBAL across rounds
    (review wf_4081ce31: a per-round index made the first call of every
    round a full read — delta2 was 60% full reads and any N>k collapsed
    to the same arm). deltaN = full read on every N-th drafter call over
    the whole generation, 1/N full-read fraction."""
    if arm == "full":
        return kv
    if arm.startswith("delta"):
        n = int(arm[len("delta"):])
        if call_idx % n == n - 1:  # every N-th call reads everything
            return kv
    k, v = kv["full_attention"]
    L = k.shape[2]
    if L <= SINK + WINDOW:
        return kv
    ksub = torch.cat([k[:, :, :SINK], k[:, :, L - WINDOW:]], dim=2)
    vsub = torch.cat([v[:, :, :SINK], v[:, :, L - WINDOW:]], dim=2)
    out = dict(kv)
    out["full_attention"] = (ksub, vsub)
    return out


@torch.no_grad()
def plain_greedy(target, ids, max_new, offload=False):
    """Reference chain via the EXACT forward pattern of run_arm — split
    prefill (bulk, then 1-token) + sequential q_len=1 decode + explicit
    hybrid cache — so every position's kernel shapes align, token 0
    included. Against this, ANY divergence is a real loop bug (review
    wf_8f9b74c6: the previous single-prefill reference could tie-flip
    token 0, and the tolerance that excused it also passed real
    divergences at >=24)."""
    from transformers import DynamicCache

    cache = make_cache(target, offload)
    target(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True,
           logits_to_keep=1)
    out = target(input_ids=ids[:, -1:], past_key_values=cache,
                 use_cache=True)
    toks = [int(out.logits[0, -1].argmax(-1).item())]
    while len(toks) < max_new:
        out = target(input_ids=torch.tensor([[toks[-1]]], device=ids.device),
                     past_key_values=cache, use_cache=True)
        toks.append(int(out.logits[0, -1].argmax(-1).item()))
    return toks


_LAST_STAGE = "?"  # which GPU call OOM'd (module-global; single-threaded)
_OOM_DUMPED = False


def _st(s):
    global _LAST_STAGE
    _LAST_STAGE = s


def make_cache(target, offload):
    from transformers import DynamicCache

    kwargs = [{"config": target.config, "offloading": offload},
              {"config": target.config}]
    for kw in kwargs:
        try:
            return DynamicCache(**kw)
        except TypeError:
            continue
    return DynamicCache(config=target.config.get_text_config())


@torch.no_grad()
def run_arm(target, assistant, embed, ids, arm, k_drafts, max_new,
            offload=False):
    """Our draft-verify loop. Returns per-prompt stats dict.

    Cache invariant: the trunk cache holds the validated prefix EXCLUDING
    the newest validated token. VERIFY IS SEQUENTIAL (one token per
    forward): q_len=1 never triggers the upstream sliding-mask/shared-KV
    length bug, so the trunk runs on its NATIVE hybrid cache
    (DynamicCache(config=...), window-capped sliding layers — the
    uncapped-cache workaround was what OOM'd the trunk prefill at >=16K
    on box 43), and stopping at the first mismatch removes cache
    cropping entirely. KNOWN 1-ROW DEVIATION from native: the native
    generator's KV slice exposes the rejected draft's KV at the bonus
    position (we never forward rejected drafts). This is a DOCUMENTED
    RESIDUAL DIFFERENCE — structurally invisible to the plain-greedy
    gate (it can only shift ACCEPTANCE, never validated tokens) — and
    only the short-context native cross-check brushes against it.
    """
    from transformers import DynamicCache

    dev = ids.device
    cache = make_cache(target, offload)
    # two-step prefill: output_hidden_states over the whole prompt would
    # materialize every layer's hidden (~22GB at 32K x 63 layers); we only
    # need the LAST position's hidden, so the bulk runs without it and a
    # 1-token step collects hidden + full-length shared KV
    _st("prefill_bulk")
    target(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True,
           logits_to_keep=1)
    _st("prefill_last")
    out = target(input_ids=ids[:, -1:], past_key_values=cache,
                 use_cache=True, output_hidden_states=True,
                 return_shared_kv_states=True)
    pending_logit = out.logits[:, -1]
    last_hidden = out.hidden_states[-1][:, -1:]
    kv = cap_sliding(out.shared_kv_states)
    validated = ids[0].tolist()
    newest = None  # newest validated token, not yet in trunk cache

    # first token is the trunk's own greedy choice
    t0 = int(pending_logit.argmax(-1).item())
    validated.append(t0)
    newest = t0

    rounds = drafted = accepted = bonus_ct = 0
    call_ms = []
    g_call = 0  # global drafter-call counter (deltaN cadence)
    while len(validated) - ids.shape[1] < max_new:
        L = len(validated)
        # kv from the last 1-token forward covers EXACTLY the L-1
        # trunk-forwarded tokens (sequential verify never forwards a
        # rejected draft) — no cropping exists or is needed; see the
        # docstring for the deliberate 1-row deviation from native
        kv_c = kv
        last_tok = torch.tensor([[newest]], device=dev)
        pos = torch.tensor([[L - 1]], dtype=torch.long, device=dev)
        drafts = []
        h = last_hidden
        t = last_tok
        for c in range(k_drafts):
            _st("draft_call")
            emb = embed(t)
            inp = torch.cat([emb, h], dim=-1)
            view = arm_view(kv_c, arm, g_call)  # OUTSIDE the timed window
            g_call += 1                         # (review wf_4081ce31)
            torch.cuda.synchronize()
            tt0 = time.monotonic()
            o = assistant(inputs_embeds=inp, attention_mask=None,
                          position_ids=pos,
                          shared_kv_states=view,
                          use_cache=False)
            torch.cuda.synchronize()
            call_ms.append((time.monotonic() - tt0) * 1000)
            t = o.logits.argmax(dim=-1)
            h = o.last_hidden_state
            drafts.append(int(t.item()))
        rounds += 1
        drafted += len(drafts)

        # SEQUENTIAL verify: forward one validated token at a time; stop
        # before ever forwarding a rejected draft (cache stays clean)
        n_match = 0
        cur = newest
        for d in drafts:
            _st("verify_step")
            vout = target(input_ids=torch.tensor([[cur]], device=dev),
                          past_key_values=cache, use_cache=True,
                          output_hidden_states=True,
                          return_shared_kv_states=True)
            nxt = int(vout.logits[0, -1].argmax(-1).item())
            last_hidden = vout.hidden_states[-1][:, -1:]
            kv = cap_sliding(vout.shared_kv_states)
            if nxt == d:
                n_match += 1
                cur = d
            else:
                bonus = nxt
                break
        else:  # every draft matched: one more step for the bonus
            _st("verify_step")
            vout = target(input_ids=torch.tensor([[cur]], device=dev),
                          past_key_values=cache, use_cache=True,
                          output_hidden_states=True,
                          return_shared_kv_states=True)
            bonus = int(vout.logits[0, -1].argmax(-1).item())
            last_hidden = vout.hidden_states[-1][:, -1:]
            kv = cap_sliding(vout.shared_kv_states)
        accepted += n_match
        bonus_ct += 1
        validated.extend(drafts[:n_match] + [bonus])
        newest = bonus

    new_tokens = validated[ids.shape[1]:]
    return dict(rounds=rounds, drafted=drafted, accepted=accepted,
                bonus=bonus_ct, total_new=len(new_tokens),
                tokens=new_tokens,
                draft_call_ms=sum(call_ms) / max(len(call_ms), 1))


def main():
    args = parse_args()
    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name="g4_mtp_g1", config=vars(args))
    import transformers
    run.config.update({"transformers_version": transformers.__version__},
                      allow_val_change=True)
    print(f"[g1] transformers {transformers.__version__}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, device_map="cuda").eval()
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=torch.bfloat16, device_map="cuda").eval()
    embed = target.get_input_embeddings()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    has_data = out_path.exists() and out_path.stat().st_size > 0
    if has_data:
        with out_path.open() as f:
            if f.readline().strip().split(",") != HEADER:
                raise SystemExit(f"CSV schema mismatch in {out_path}")
    fh = out_path.open("a", newline="")
    w = csv.writer(fh)
    if not has_data:
        w.writerow(HEADER)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    if args.parity_check:
        # independent cross-check vs NATIVE assisted decoding, once per
        # run, at a context SHORTER than the sliding window (semantics
        # coincide across cache types there) — guards the shared-bug
        # blindness of a self-referential plain-greedy gate (wf_8f9b74c6
        # finding 3). Batched-verify kernel shapes still differ, so the
        # threshold tolerates late tie-flips but not systematic breakage.
        from transformers import DynamicCache
        ctx = load_contexts(tok, 1, 800)[0]
        sp = tok.apply_chat_template(
            [{"role": "user", "content": INSTR + ctx}],
            add_generation_prompt=True, tokenize=False)
        ids_s = tok(sp, return_tensors="pt",
                    add_special_tokens=False)["input_ids"].cuda()
        r_full = run_arm(target, assistant, embed, ids_s, "full", args.k,
                         32, offload=args.offload)
        with torch.no_grad():
            nat = target.generate(input_ids=ids_s, max_new_tokens=32,
                                  do_sample=False, assistant_model=assistant,
                                  past_key_values=DynamicCache())
        nat_new = nat[0, ids_s.shape[1]:].tolist()
        m = 0
        for a, b in zip(nat_new, r_full["tokens"]):
            if a != b:
                break
            m += 1
        n_cmp = min(len(nat_new), len(r_full["tokens"]))
        print(f"[g1] native cross-check (short ctx): prefix {m}/{n_cmp}",
              flush=True)
        run.summary["native_crosscheck_prefix"] = m
        if m < min(16, n_cmp):
            raise SystemExit(
                f"G1 GATE FAILED: short-context native cross-check "
                f"diverges at token {m} — loop deviates from the shipped "
                "assisted-decoding behavior even where cache semantics "
                "coincide")

    for tier in [int(x) for x in args.tiers.split(",")]:
        ctxs = load_contexts(tok, args.n, tier)
        for i, ctx in enumerate(ctxs):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": INSTR + ctx}],
                add_generation_prompt=True, tokenize=False)
            ids = tok(prompt, return_tensors="pt",
                      add_special_tokens=False)["input_ids"].cuda()
            ref_tokens = None
            for arm in arms:
                try:
                    r = run_arm(target, assistant, embed, ids, arm,
                                args.k, args.max_new,
                                offload=args.offload)
                except torch.cuda.OutOfMemoryError as e:
                    w.writerow([tier, i, arm, ids.shape[1],
                                f"OOM@{_LAST_STAGE}"] +
                               [""] * (len(HEADER) - 6) + [run.id])
                    fh.flush()
                    print(f"[g1] tier {tier} #{i} {arm}: OOM at stage "
                          f"{_LAST_STAGE}: {str(e)[:200]}", flush=True)
                    global _OOM_DUMPED
                    if not _OOM_DUMPED:
                        _OOM_DUMPED = True
                        import traceback
                        traceback.print_exc()
                        print(torch.cuda.memory_summary(abbreviated=True),
                              flush=True)
                    oomed = True
                else:
                    oomed = False
                if oomed:
                    # empty_cache INSIDE the handler ran while the live
                    # exception still pinned run_arm's multi-GB locals —
                    # cleanup only works after the frame is released
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                import gc
                gc.collect()
                torch.cuda.empty_cache()  # 48 arm-runs of allocator litter
                # contributed to the 16K wall (59GB non-releasable, box 44)
                if arm == "full":
                    ref_tokens = r["tokens"]
                # every validated token is a trunk argmax, so all arms
                # must share a common prefix BY CONSTRUCTION — this
                # column is a bug invariant, not an output-quality
                # metric (review wf_4081ce31; lengths differ benignly
                # with round-boundary overshoot)
                if ref_tokens is None or arm == "full":
                    match = ""
                else:
                    m = min(len(r["tokens"]), len(ref_tokens))
                    match = int(r["tokens"][:m] == ref_tokens[:m])
                apr = r["accepted"] / max(r["rounds"], 1)
                rate = r["accepted"] / max(r["drafted"], 1)
                w.writerow([tier, i, arm, ids.shape[1], r["rounds"],
                            r["drafted"], r["accepted"], f"{apr:.3f}",
                            f"{rate:.3f}", r["bonus"], r["total_new"],
                            f"{r['draft_call_ms']:.2f}", match, run.id])
                fh.flush()
                print(f"[g1] tier {tier} #{i} {arm}: acc/round {apr:.2f} "
                      f"rate {rate:.2f} draft-call {r['draft_call_ms']:.1f}ms",
                      flush=True)

            if args.parity_check:
                if ref_tokens is None:  # BEFORE the expensive reference
                    raise SystemExit(
                        "G1 GATE MISCONFIGURED: --parity-check needs a "
                        "successful 'full' arm run (it OOM'd or --arms "
                        "omits it) — cannot certify")
                ref2 = plain_greedy(target, ids, args.max_new,
                                    offload=args.offload)
                n_cmp = min(len(ref2), len(ref_tokens))
                m = 0
                for a, b in zip(ref2, ref_tokens):
                    if a != b:
                        break
                    m += 1
                print(f"[g1] parity vs plain-greedy (shape-aligned): "
                      f"prefix {m}/{n_cmp}", flush=True)
                run.summary[f"parity_prefix_t{tier}_p{i}"] = m
                if m < n_cmp:  # ANY divergence is a loop bug — the
                    raise SystemExit(  # reference is shape-identical
                        "G1 GATE FAILED: our loop diverges from PLAIN "
                        f"trunk greedy at token {m}/{n_cmp} — loop bug")
    fh.close()
    run.finish()
    print("[g1] DONE", flush=True)


if __name__ == "__main__":
    main()
