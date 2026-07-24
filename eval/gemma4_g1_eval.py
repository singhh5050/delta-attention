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
    deltaN   — sparse, but every N-th drafter call reads the full entry
               (anchor-refresh cadence, global counter)
    deltacorrN — THE DELTA CORRECTION: sparse reads, but round-call 0
               (and every N-th call after, within the round) anchors —
               the drafter runs on both views, delta = full - sparse at
               the full-attention layer output (captured via forward
               hook; post-o_proj == pre-o_proj correction by linearity),
               and delta is ADDED to subsequent sparse calls' outputs.
               deltacorr5 with k=5 -> one anchor per round.
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
import gc
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
          "draft_call_ms", "draft_call_ms_warm", "match_vs_full", "run_id"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--tiers", type=str, default="4096,16384,32768")
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--k", type=int, default=5, help="drafts per round")
    p.add_argument("--arms", type=str, default="full,sparse,delta4")
    p.add_argument("--parity-check", action="store_true",
                   help="zero-tolerance parity gate: the full arm must match "
                        "PLAIN trunk greedy (shape-aligned reference) at "
                        "every tier, plus one short-context native-assisted "
                        "cross-check per run")
    p.add_argument("--target", type=str, default=TARGET)
    p.add_argument("--assistant", type=str, default=ASSISTANT)
    p.add_argument("--offload", action="store_true",
                   help="CPU-offloaded trunk cache. BROKEN UPSTREAM as of "
                        "transformers 5.15.0.dev0: nondeterministic + "
                        "degenerate trunk outputs once ctx > ~2K (box-45 "
                        "diag T1, 2026-07-22 — two identical plain chains "
                        "diverge at token 0). Kept for when upstream fixes "
                        "it; do not use for results")
    p.add_argument("--prefill-chunk", type=int, default=0,
                   help="bulk-prefill in chunks of this many tokens (0 = "
                        "single shot). The >=16K OOM was prefill "
                        "TRANSIENTS, not cache (hybrid cache at 65K is "
                        "~5GB vs 62GB weights); chunking caps transients "
                        "on the NATIVE cache instead of offloading")
    p.add_argument("--chunk-equiv-check", action="store_true",
                   help="once per tier (first prompt): chunked and "
                        "single-shot prefill must agree at the SOURCE — "
                        "same greedy argmax, last-position logits and "
                        "shared-KV rows equal to numeric noise (token "
                        "chains are NOT compared: different kernel shapes "
                        "tie-flip legitimately). Certifies the chunked "
                        "continuation path (the upstream sliding-mask bug "
                        "class lives there); only runs usefully at tiers "
                        "that fit single-shot (4K/8K)")
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


def shared_kv_from_cache(target, cache):
    """Rebuild shared_kv_states by reading the cache's store_full_length_kv
    layers AFTER the forward. Under an offloaded cache, passing
    return_shared_kv_states=True corrupts the TRUNK FORWARD ITSELF once
    ctx > sliding window (box-45 diagnostic 2026-07-22: greedy chain
    diverges after 4 tokens at ctx 2000 and degenerates; offload alone and
    offload+output_hidden_states are token-exact — upstream transformers
    bug, flag leaks into layer kwargs). The dict the model would return is
    exactly the post-update states of the last non-shared layer per type,
    which the cache still holds — so read them out-of-band instead."""
    torch.cuda.synchronize()  # offload D2H copies run on a side stream
    mdl = target
    for _ in range(4):  # unwrap ForCausalLM/multimodal shells to the decoder
        if hasattr(mdl, "layers"):
            break
        mdl = (mdl.language_model if hasattr(mdl, "language_model")
               else mdl.model)
    else:
        raise SystemExit("shared_kv_from_cache: no decoder .layers found")
    out = {}
    for i, layer in enumerate(mdl.layers):
        attn = layer.self_attn
        if getattr(attn, "store_full_length_kv", False):
            lay = cache.layers[i]
            out[attn.layer_type] = (lay.keys, lay.values)
    if set(out) != {"full_attention", "sliding_attention"}:
        raise SystemExit(f"shared_kv_from_cache: expected one store layer "
                         f"per type, got {sorted(out)}")
    return out


def kv_to_dev(kv, dev):
    """Materialize a shared_kv_states view on the compute device. The
    drafter's own forward would do this move INSIDE our timed window
    (modeling_gemma4_assistant moves shared_kv_states to target_device),
    which under --offload charges the full arm a whole-cache H2D copy per
    call while sparse pays a tiny one — a PCIe artifact, not read cost.
    No-op (returns the same tensors) when the view is already resident."""
    return {k_: (kk.to(dev, non_blocking=True), vv.to(dev, non_blocking=True))
            for k_, (kk, vv) in kv.items()}


import re as _re

ARM_RE = _re.compile(r"^(full|sparse|delta[1-9][0-9]*|deltacorr[1-9][0-9]*)$")


def validate_arms(arms):
    """Fail fast on unknown arm names: arm_view used to treat ANY
    unrecognized string as the sparse arm, so a typo ('detla4') silently
    produced plausible-looking rows under a fake label (review a15f950a
    finding 2)."""
    bad = [a for a in arms if not ARM_RE.match(a)]
    if bad:
        raise SystemExit(
            f"unknown arm(s) {bad}; valid: full | sparse | deltaN "
            "(full read every Nth call) | deltacorrN (sparse reads with "
            "the delta CORRECTION, anchor every Nth call per round)")


def sparse_view(kv):
    """sink+window subsample of the full_attention entry (StreamingLLM
    read; rope is baked into the trunk's cached keys, so gathering
    preserves positions). No-op when the prefix fits inside sink+window."""
    k, v = kv["full_attention"]
    L = k.shape[2]
    if L <= SINK + WINDOW:
        return kv
    ksub = torch.cat([k[:, :, :SINK], k[:, :, L - WINDOW:]], dim=2)
    vsub = torch.cat([v[:, :, :SINK], v[:, :, L - WINDOW:]], dim=2)
    out = dict(kv)
    out["full_attention"] = (ksub, vsub)
    return out


def arm_view(kv, arm, call_idx):
    """The intervention: how THIS drafter call sees the trunk KV (already
    exactly the validated prefix). call_idx is GLOBAL across rounds
    (review wf_4081ce31: a per-round index made the first call of every
    round a full read — delta2 was 60% full reads and any N>k collapsed
    to the same arm). deltaN = full read on every N-th drafter call over
    the whole generation, 1/N full-read fraction. deltacorrN is handled
    in the draft loop (it needs BOTH views at anchor calls)."""
    if arm == "full":
        return kv
    if arm.startswith("delta") and not arm.startswith("deltacorr"):
        n = int(arm[len("delta"):])
        if call_idx % n == n - 1:  # every N-th call reads everything
            return kv
    return sparse_view(kv)


class DeltaCorrState:
    """Output-space delta correction for the drafter's full-attention
    layer(s), via forward hooks — THE DRAFTER IS NEVER MODIFIED; the hook
    only reads or adds to the module's output tensor.

    Mechanism (the correction we previously skipped): at an ANCHOR call
    the drafter runs twice — once on the full KV view (its output is the
    draft), once on the sparse view (bookkeeping) — and the hook captures
    the full-attention module's output under each; delta = full - sparse.
    On the following sparse calls the hook ADDS delta to the module
    output. The module output is post-o_proj; o_proj is linear, so
    correcting there is mathematically identical to the paper's
    pre-o_proj correction (the projection of the difference equals the
    difference of the projections; bias cancels in the subtraction).

    Cadence is PER ROUND, anchor on round-call 0 by default: within a
    drafting round the KV view is frozen (only the query changes), so
    delta staleness is pure query drift — the well-posed regime, unlike
    decode-time correction over a growing cache."""

    def __init__(self):
        self.mode = "off"  # off | capture | apply
        self.buf = {}
        self.delta = {}

    def hook(self, module, args, kwargs, output):
        is_tuple = isinstance(output, tuple)
        out0 = output[0] if is_tuple else output
        if self.mode == "capture":
            self.buf[id(module)] = out0
            return None
        if self.mode == "apply" and id(module) in self.delta:
            corrected = out0 + self.delta[id(module)]
            return ((corrected,) + tuple(output[1:])) if is_tuple else corrected
        return None


def full_attn_modules(assistant):
    """The drafter's full_attention self-attn module(s), located via
    config.layer_types — no hardcoded layer index."""
    mdl = assistant
    for _ in range(4):
        if hasattr(mdl, "layers"):
            break
        mdl = (mdl.language_model if hasattr(mdl, "language_model")
               else mdl.model)
    else:
        raise SystemExit("deltacorr: no decoder .layers found on assistant")
    lts = assistant.config.get_text_config().layer_types
    mods = [mdl.layers[i].self_attn for i, t in enumerate(lts)
            if t == "full_attention"]
    if not mods:
        raise SystemExit("deltacorr: assistant has no full_attention layer")
    return mods


def bulk_prefill(target, ids_prefix, cache, chunk):
    """Prefill all but the last prompt token, optionally in chunks to cap
    transient memory (positions derive from cache length, so chunk
    boundaries are transparent to the model)."""
    L = ids_prefix.shape[1]
    step = chunk if chunk and chunk > 0 else L
    for s in range(0, L, step):
        target(input_ids=ids_prefix[:, s:s + step], past_key_values=cache,
               use_cache=True, logits_to_keep=1)


@torch.no_grad()
def plain_greedy(target, ids, max_new, offload=False, chunk=0):
    """Reference chain via the EXACT forward pattern of run_arm — split
    prefill (bulk, then 1-token) + sequential q_len=1 decode + the same
    make_cache — so every position's kernel shapes align, token 0
    included. Against this, ANY divergence is a real loop bug (review
    wf_8f9b74c6: the previous single-prefill reference could tie-flip
    token 0, and the tolerance that excused it also passed real
    divergences at >=24)."""
    cache = make_cache(target, offload)
    bulk_prefill(target, ids[:, :-1], cache, chunk)
    # SAME kwargs as run_arm's forwards via the shared trunk_kwargs owner
    # — including the OFFLOAD GATING of return_shared_kv_states (a
    # hand-copied always-on flag here ran the documented corruption path
    # in the reference while the loop omitted it, so the gate blamed the
    # loop; review a15f950a finding 1)
    kw = trunk_kwargs(offload)
    out = target(input_ids=ids[:, -1:], past_key_values=cache, **kw)
    toks = [int(out.logits[0, -1].argmax(-1).item())]
    while len(toks) < max_new:
        out = target(input_ids=torch.tensor([[toks[-1]]], device=ids.device),
                     past_key_values=cache, **kw)
        toks.append(int(out.logits[0, -1].argmax(-1).item()))
    return toks


_LAST_STAGE = "?"  # which GPU call OOM'd (module-global; single-threaded)
_OOM_DUMPED = False


def _st(s):
    global _LAST_STAGE
    _LAST_STAGE = s


def trunk_kwargs(offload):
    """THE single owner of trunk-forward kwargs. Five call sites drifted
    twice under hand-copying (reviews wf_8f9b74c6/wf_31d5f03b, then the
    offload gating desync): under --offload, return_shared_kv_states
    corrupts the trunk forward itself (see shared_kv_from_cache), so the
    flag is gated identically for the LOOP and its parity REFERENCE."""
    kw = {"use_cache": True, "output_hidden_states": True}
    if not offload:
        kw["return_shared_kv_states"] = True
    return kw


def make_cache(target, offload):
    from transformers import DynamicCache

    if offload:
        # NEVER fall back silently: a non-offloaded hybrid cache is exactly
        # the >=16K OOM wall (76GB at 16K prefill, box 44), and a silent
        # fallback would also mislabel the cache config of every result
        try:
            return DynamicCache(config=target.config, offloading=True)
        except TypeError as e:
            raise SystemExit(
                f"--offload requested but DynamicCache rejects offloading= "
                f"({e}) — installed transformers-main drifted; refusing to "
                "run non-offloaded under an offload label")
    try:
        return DynamicCache(config=target.config)
    except TypeError:
        return DynamicCache(config=target.config.get_text_config())


@torch.no_grad()
def run_arm(target, assistant, embed, ids, arm, k_drafts, max_new,
            offload=False, chunk=0):
    """Our draft-verify loop. Returns per-prompt stats dict.

    Cache invariant: the trunk cache holds the validated prefix EXCLUDING
    the newest validated token. VERIFY IS SEQUENTIAL (one token per
    forward): q_len=1 never triggers the upstream sliding-mask/shared-KV
    length bug, and stopping at the first mismatch removes cache cropping
    entirely. Cache: without --offload the trunk runs on its NATIVE hybrid
    cache (DynamicCache(config=...), window-capped sliding layers); with
    --offload it runs on an UNCAPPED CPU-offloaded cache (the hybrid cache
    allocates 76GB at a 16K prefill, box 44) whose full-length sliding
    shared-KV entry we re-cap to native length (window+1) via cap_sliding
    before the drafter ever sees it. KNOWN 1-ROW DEVIATION from native: the native
    generator's KV slice exposes the rejected draft's KV at the bonus
    position (we never forward rejected drafts). This is a DOCUMENTED
    RESIDUAL DIFFERENCE — structurally invisible to the plain-greedy
    gate (it can only shift ACCEPTANCE, never validated tokens) — and
    only the short-context native cross-check brushes against it.
    """
    dev = ids.device
    cache = make_cache(target, offload)
    kw = trunk_kwargs(offload)  # owns the offload kvflag gating

    def grab_kv(o):
        return cap_sliding(shared_kv_from_cache(target, cache) if offload
                           else o.shared_kv_states)

    # two-step prefill: output_hidden_states over the whole prompt would
    # materialize every layer's hidden (~22GB at 32K x 63 layers); we only
    # need the LAST position's hidden, so the bulk runs without it and a
    # 1-token step collects hidden + full-length shared KV
    _st("prefill_bulk")
    bulk_prefill(target, ids[:, :-1], cache, chunk)
    _st("prefill_last")
    out = target(input_ids=ids[:, -1:], past_key_values=cache, **kw)
    pending_logit = out.logits[:, -1]
    last_hidden = out.hidden_states[-1][:, -1:]
    kv = grab_kv(out)
    validated = ids[0].tolist()
    newest = None  # newest validated token, not yet in trunk cache

    # first token is the trunk's own greedy choice
    t0 = int(pending_logit.argmax(-1).item())
    validated.append(t0)
    newest = t0

    rounds = drafted = accepted = bonus_ct = 0
    call_ms = []
    g_call = 0  # global drafter-call counter (deltaN cadence)
    corr = None
    corr_n = 0
    hooks = []
    if arm.startswith("deltacorr"):
        corr = DeltaCorrState()
        corr_n = int(arm[len("deltacorr"):])
        hooks = [m.register_forward_hook(corr.hook, with_kwargs=True)
                 for m in full_attn_modules(assistant)]
    while len(validated) - ids.shape[1] < max_new:
        L = len(validated)
        # kv from the last 1-token forward covers EXACTLY the L-1
        # trunk-forwarded tokens (sequential verify never forwards a
        # rejected draft) — no cropping exists or is needed; see the
        # docstring for the deliberate 1-row deviation from native
        kv_c = kv
        moved_full = None  # per-round memo: the full read's on-device copy
        last_tok = torch.tensor([[newest]], device=dev)
        pos = torch.tensor([[L - 1]], dtype=torch.long, device=dev)
        drafts = []
        h = last_hidden
        t = last_tok
        moved_sparse = None  # per-round memo (kv frozen within a round)
        for c in range(k_drafts):
            _st("draft_call")
            emb = embed(t)
            inp = torch.cat([emb, h], dim=-1)
            if corr is not None:
                # deltacorr: cadence is PER ROUND (kv frozen within a
                # round -> delta staleness is pure query drift); anchor
                # on round-call 0 and every corr_n-th call after
                if moved_sparse is None:
                    moved_sparse = kv_to_dev(sparse_view(kv_c), dev)
                anchor = (c % corr_n == 0)
                if anchor and moved_full is None:
                    moved_full = kv_to_dev(kv_c, dev)
                view = moved_full if anchor else moved_sparse
            else:
                view = arm_view(kv_c, arm, g_call)  # OUTSIDE timed window
                if view is kv_c:  # full read: identical each call
                    if moved_full is None:
                        moved_full = kv_to_dev(view, dev)
                    view = moved_full
                else:
                    view = kv_to_dev(view, dev)
            g_call += 1                             # (review wf_4081ce31)
            torch.cuda.synchronize()
            tt0 = time.monotonic()
            if corr is not None:
                corr.mode = "apply" if not anchor else "off"
            o = assistant(inputs_embeds=inp, attention_mask=None,
                          position_ids=pos,
                          shared_kv_states=view,
                          use_cache=False)
            torch.cuda.synchronize()
            call_ms.append((time.monotonic() - tt0) * 1000)
            if corr is not None and anchor:
                # bookkeeping (UNtimed — the timed call above produced the
                # draft): capture full-attn outputs under full then sparse
                # view for the SAME input, delta = full - sparse
                corr.mode = "capture"
                corr.buf = {}
                assistant(inputs_embeds=inp, attention_mask=None,
                          position_ids=pos, shared_kv_states=moved_full,
                          use_cache=False)
                full_buf = corr.buf
                corr.buf = {}
                assistant(inputs_embeds=inp, attention_mask=None,
                          position_ids=pos, shared_kv_states=moved_sparse,
                          use_cache=False)
                corr.delta = {mid: full_buf[mid] - corr.buf[mid]
                              for mid in full_buf}
                corr.mode = "off"
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
                          past_key_values=cache, **kw)
            nxt = int(vout.logits[0, -1].argmax(-1).item())
            last_hidden = vout.hidden_states[-1][:, -1:]
            kv = grab_kv(vout)
            if nxt == d:
                n_match += 1
                cur = d
            else:
                bonus = nxt
                break
        else:  # every draft matched: one more step for the bonus
            _st("verify_step")
            vout = target(input_ids=torch.tensor([[cur]], device=dev),
                          past_key_values=cache, **kw)
            bonus = int(vout.logits[0, -1].argmax(-1).item())
            last_hidden = vout.hidden_states[-1][:, -1:]
            kv = grab_kv(vout)
        accepted += n_match
        bonus_ct += 1
        validated.extend(drafts[:n_match] + [bonus])
        newest = bonus

    for hk in hooks:
        hk.remove()
    new_tokens = validated[ids.shape[1]:]
    # warm mean drops the first timed call: after the inter-arm
    # empty_cache every arm's call 0 pays allocator-pool growth (and
    # prompt 0's first arm pays one-time kernel autotune) — report both
    warm = call_ms[1:] if len(call_ms) > 1 else call_ms
    return dict(rounds=rounds, drafted=drafted, accepted=accepted,
                bonus=bonus_ct, total_new=len(new_tokens),
                tokens=new_tokens,
                draft_call_ms=sum(call_ms) / max(len(call_ms), 1),
                draft_call_ms_warm=sum(warm) / max(len(warm), 1))


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

    # derive the native sliding-entry cap from the ACTUAL config instead of
    # trusting the hand-derived 1025 (window+1): a different --target or an
    # upstream window change would silently re-create the box-42/43
    # sliding-semantics artifact class
    sw = getattr(target.config.get_text_config(), "sliding_window", None)
    if sw:
        globals()["SLIDING_CAP"] = sw + 1
    print(f"[g1] sliding cap {SLIDING_CAP} (config sliding_window={sw}), "
          f"offload={args.offload}", flush=True)
    run.summary["sliding_cap"] = SLIDING_CAP

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

    def status_row(tier_, i_, arm_, ptoks, status):
        # single owner of the non-numeric row shape: the padded template
        # previously lived in two copies (review a15f950a finding 7)
        w.writerow([tier_, i_, arm_, ptoks, status] +
                   [""] * (len(HEADER) - 6) + [run.id])
        fh.flush()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    validate_arms(arms)  # typos used to silently run as 'sparse'
    if "full" in arms:  # pairing anchor must run first — every other
        arms.remove("full")  # arm's rows are interpreted against it
        arms.insert(0, "full")

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
                         32, offload=args.offload, chunk=args.prefill_chunk)
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
        if n_cmp < 16:
            raise SystemExit(
                f"G1 GATE MISCONFIGURED: native cross-check compared only "
                f"{n_cmp} tokens (early EOS?) — too short to certify "
                "anything; pick a different context or raise max_new")
        if m < min(16, n_cmp):
            raise SystemExit(
                f"G1 GATE FAILED: short-context native cross-check "
                f"diverges at token {m} — loop deviates from the shipped "
                "assisted-decoding behavior even where cache semantics "
                "coincide")

    for tier in [int(x) for x in args.tiers.split(",")]:
        ctxs = load_contexts(tok, args.n, tier)
        chunk_certified = False
        for i, ctx in enumerate(ctxs):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": INSTR + ctx}],
                add_generation_prompt=True, tokenize=False)
            ids = tok(prompt, return_tensors="pt",
                      add_special_tokens=False)["input_ids"].cuda()

            if (args.chunk_equiv_check and args.prefill_chunk
                    and not chunk_certified):
                # The chunk-continuation forward (q_len>1 over a past
                # sliding cache) is the upstream sliding-mask bug zone,
                # and both the loop AND its parity reference chunk — a
                # broken continuation would be invisible to the gate.
                # Token-exact equality vs single-shot is NOT a valid
                # invariant here (different kernel shapes => bf16
                # tie-flips; first attempt matched 38/64 then flipped —
                # the specdec "parity is not bitwise" lesson). The valid
                # invariant is at the SOURCE: after both prefills, the
                # shared-KV cache rows and last-position logits must
                # agree to numeric noise, and the greedy argmax must
                # match. Mask corruption moves logits by many units and
                # KV rows by O(signal); reduction-order noise does not.
                sides = {}
                with torch.no_grad():
                    for label, ch in (("chunked", args.prefill_chunk),
                                      ("single", 0)):
                        cache = make_cache(target, False)
                        bulk_prefill(target, ids[:, :-1], cache, ch)
                        o = target(input_ids=ids[:, -1:],
                                   past_key_values=cache, use_cache=True)
                        sides[label] = (shared_kv_from_cache(target, cache),
                                        o.logits[0, -1].float().cpu())
                        del cache, o
                        gc.collect()
                        torch.cuda.empty_cache()
                (kv_c, lg_c), (kv_s, lg_s) = sides["chunked"], sides["single"]
                kv_rel = 0.0
                for t in kv_c:
                    for a, b in zip(kv_c[t], kv_s[t]):
                        n = min(a.shape[2], b.shape[2])
                        d = (a[:, :, -n:].float()
                             - b[:, :, -n:].float()).abs().max().item()
                        kv_rel = max(kv_rel, d / max(b.float().std().item(),
                                                     1e-6))
                logit_d = (lg_c - lg_s).abs().max().item()
                argmax_same = int(lg_c.argmax().item() == lg_s.argmax().item())
                print(f"[g1] chunk-equiv (tier {tier}): argmax_same="
                      f"{argmax_same} max|dlogit|={logit_d:.3f} "
                      f"kv_rel={kv_rel:.4f}", flush=True)
                run.summary[f"chunk_equiv_t{tier}"] = dict(
                    argmax_same=argmax_same, logit_d=logit_d, kv_rel=kv_rel)
                if not argmax_same or logit_d > 2.0 or kv_rel > 0.5:
                    raise SystemExit(
                        f"G1 GATE FAILED: chunked prefill (chunk="
                        f"{args.prefill_chunk}) is not numerically "
                        f"equivalent to single-shot (argmax_same="
                        f"{argmax_same}, max|dlogit|={logit_d:.3f}, "
                        f"kv_rel={kv_rel:.4f}) — chunk-continuation path "
                        "is broken; results at chunked tiers would be "
                        "wrong")
                chunk_certified = True

            ref_tokens = None
            full_oomed = False
            prompt_oomed = False
            for arm in arms:
                try:
                    r = run_arm(target, assistant, embed, ids, arm,
                                args.k, args.max_new,
                                offload=args.offload,
                                chunk=args.prefill_chunk)
                except (torch.cuda.OutOfMemoryError, MemoryError,
                        RuntimeError) as e:
                    # --offload can fail host-side (pinned/pageable alloc),
                    # surfacing as MemoryError or a RuntimeError — treat
                    # those as OOM rows too, but never mask a real bug
                    if (isinstance(e, RuntimeError)
                            and not isinstance(e, torch.cuda.OutOfMemoryError)
                            and "out of memory" not in str(e).lower()):
                        raise
                    status_row(tier, i, arm, ids.shape[1],
                               f"OOM@{_LAST_STAGE}")
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
                    if arm == "full":
                        full_oomed = True
                else:
                    oomed = False
                # ONE cleanup for both paths, AFTER the except frame is
                # released (empty_cache inside the handler ran while the
                # live exception still pinned run_arm's multi-GB locals);
                # 48 arm-runs of allocator litter contributed to the 16K
                # wall (59GB non-releasable, box 44)
                gc.collect()
                torch.cuda.empty_cache()
                if oomed:
                    prompt_oomed = True
                    if arm == "full":
                        # no pairing anchor for this prompt: the other
                        # arms' rows would enter per-arm means UNPAIRED
                        # and silently skew the cross-arm comparison
                        # (review wf_31d5f03b finding 7) — skip them,
                        # visibly
                        for a2 in arms[arms.index(arm) + 1:]:
                            status_row(tier, i, a2, ids.shape[1],
                                       "SKIPPED-no-full-pair")
                        print(f"[g1] tier {tier} #{i}: remaining arms "
                              "SKIPPED (full arm OOM'd — no pair)",
                              flush=True)
                        break
                    continue
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
                            f"{r['draft_call_ms']:.2f}",
                            f"{r['draft_call_ms_warm']:.2f}", match, run.id])
                fh.flush()
                print(f"[g1] tier {tier} #{i} {arm}: acc/round {apr:.2f} "
                      f"rate {rate:.2f} draft-call {r['draft_call_ms']:.1f}ms",
                      flush=True)

            if prompt_oomed and not full_oomed:
                # a NON-full arm OOM'd after full's row was flushed: mark
                # the prompt so aggregation excludes it — one-directional
                # pairing left full's mean including prompts the other
                # arms' means excluded (review a15f950a finding 3).
                # Aggregate rule: drop any (tier, idx) with a marker row.
                status_row(tier, i, "PROMPT-UNPAIRED", ids.shape[1],
                           "drop this prompt from per-arm means")
                print(f"[g1] tier {tier} #{i}: marked UNPAIRED (a non-full "
                      "arm OOM'd)", flush=True)

            if args.parity_check:
                if ref_tokens is None:  # BEFORE the expensive reference
                    if full_oomed:
                        # nothing to certify for this prompt (its OOM row
                        # is already written) — skip, don't kill the chain
                        print(f"[g1] tier {tier} #{i}: parity SKIPPED — "
                              "full arm OOM'd, no result to certify",
                              flush=True)
                        run.summary[f"parity_prefix_t{tier}_p{i}"] = "oom"
                        continue
                    raise SystemExit(
                        "G1 GATE MISCONFIGURED: --parity-check needs the "
                        "'full' arm in --arms — cannot certify")
                ref2 = plain_greedy(target, ids, args.max_new,
                                    offload=args.offload,
                                    chunk=args.prefill_chunk)
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
