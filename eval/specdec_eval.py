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
# draft weights: base model or a trained adapter (merged at init).
# Paths resolve against REPO_ROOT so the harness is cwd-independent.
DRAFT_WEIGHTS = {"base": "",
                 "ce32k": str(REPO_ROOT / "checkpoints/pilot_delta_32k"),
                 "dft": str(REPO_ROOT / "checkpoints/pilot_distill_dft"),
                 "dftmix": str(REPO_ROOT / "checkpoints/pilot_distill_dftmix")}


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


TIE_EPS = 0.5  # bf16 ulp at |logit|~16-32 is 0.125-0.25: a kernel-order
# argmax flip needs the two candidates within ~1-2 ulps, while a harness bug
# puts the spec token several logits below the max. Measured flips on
# GovReport (margin probe, 07-16): 0.000, 0.125, 0.250.


def classify_parity(entries, min_prefix):
    """entries: per-checked-prompt parity results — int (divergence index;
    10**9 = certified byte-identical; -1 = length mismatch after a clean
    prefix, always a bug) or ('tie', div, margin) for an early divergence
    measured as a bf16 argmax near-tie under the verify-path probe.
    Returns (csv_value, failure_msg_or_None).

    Guarantees (restored/tightened after the 07-16 review): -1 fails even
    with the gate disabled; ties are benign only with corroboration — at
    least one checked prompt must certify >= min_prefix tokens, otherwise
    an all-early-ties config would pass with ~nothing verified."""
    if not entries:
        return None, None
    nums = [e for e in entries if isinstance(e, int)]
    ties = [e for e in entries if not isinstance(e, int)]
    if -1 in nums:
        return -1, ("length mismatch after a byte-identical prefix on a "
                    "checked prompt — always a harness bug (a legit eos stop "
                    "matches through the eos, so lengths agree)")
    parity = min(nums) if nums else None
    val = "full" if parity == 10**9 else parity
    fail = None
    if min_prefix:
        if isinstance(val, int) and val < min_prefix:
            fail = (f"diverged from dense greedy at token {val} "
                    f"(< {min_prefix}) with a non-tie margin — harness bug, "
                    "not kernel numerics")
        elif ties:
            certified = [e for e in nums if e >= min_prefix] \
                + [t for t in ties if t[1] >= min_prefix]
            if not certified:
                fail = ("every checked prompt tie-flipped before the "
                        f"{min_prefix}-token gate — ties are only benign "
                        "with at least one prompt certified past the gate; "
                        "nothing verified here")
    if val is None:
        val = "tie-only"  # every checked prompt hit a measured tie-flip
    if ties:
        val = (f"{val}+{len(ties)}tie("
               + ",".join(f"{d}@{m:.2f}" for _, d, m in ties) + ")")
    return val, fail


def spec_token_margin(model, tokenizer, prompt, ref, div, spec_tok):
    """max_logit - logit[spec_tok] at the divergence slot, computed under
    the TARGET semantics: pipeline prefill of the prompt, then ref[:div] as
    ONE dense rectangle (the verify path). The tie detector for early
    divergences: ~0 (within TIE_EPS) for kernel-order coin flips, several
    logits for real bugs. NOTE: a single no-cache forward under
    attn_implementation_original would dispatch to delta_forward — the
    APPROXIMATE window+delta path — which is the wrong adjudicator (07-16
    review); this reproduces what the dense arm actually computes."""
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False)["input_ids"][0].tolist()
    setattr(model, "no_lm_head", True)
    _reset_dec_state(model)
    model.config._attn_implementation = model.config.attn_implementation_original
    with torch.no_grad():
        out = model(torch.tensor([ids], device="cuda"), use_cache=True)
        logits = model.lm_head(out.logits[:, -1, :]).float()[0]
    model.config._attn_implementation = "sdpa_rectangle"
    model.config.decode_mode = "dense"
    if div > 0:
        logits, _ = _forward_tokens(model, ref[:div], out.past_key_values)
        logits = logits[-1]
    return float(logits.max() - logits[spec_tok])


def positional_stats(nacc, block):
    """From per-block accepted-prefix lengths, compute (pos_acc, genuine).
    Prefix acceptance is monotone, so P(position i accepted) = fraction of
    blocks with n_acc > i. `genuine` is acceptance over positions 2..K only —
    the delta draft's position 1 comes from a dense anchor row and is
    accepted ~always by construction, so it is excluded for comparability
    with small-draft speculative-decoding literature."""
    nb = max(len(nacc), 1)
    pos_acc = [sum(1 for a in nacc if a > i) / nb for i in range(block)]
    if block == 1:
        return pos_acc, pos_acc[0]
    genuine = (sum(max(0, min(a, block) - 1) for a in nacc)
               / (nb * (block - 1)))
    return pos_acc, genuine


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
    # the draft's per-phase state reset means it must anchor ONLY at block
    # starts — a small config.gamma_dec would fire mid-block for large K
    model.config.gamma_dec = 10**9
    cache = out.past_key_values
    torch.cuda.synchronize()  # prefill kernels are async; time them here,
    t_prefill = time.monotonic() - t0  # not inside the first draft window

    L_dense = len(ids)                      # dense-grade cache length
    pending = [int(first_logits.argmax())]  # accepted, KV not yet in cache
    generated = [pending[0]]
    stats = {"proposed": 0, "accepted": 0, "verify_calls": 0, "draft_calls": 0,
             "full_blocks": 0, "blocks": 0, "first_reject_pos": [], "nacc": [],
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
        torch.cuda.synchronize()  # timers must include the queued GPU work
        stats["t_draft"] += time.monotonic() - t0

        # ---- verify: crop to dense-grade, one rectangle block ----
        model.config.decode_mode = "dense"
        cache.crop(L_dense)
        t0 = time.monotonic()
        blk = pending + proposals
        logits, cache = _forward_tokens(model, blk, cache)
        stats["verify_calls"] += 1
        torch.cuda.synchronize()  # timers must include the queued GPU work
        stats["t_verify"] += time.monotonic() - t0
        # dense choice at proposals[i]'s slot = argmax after its predecessor
        m = len(pending)
        dense_choices = [int(logits[m - 1 + i].argmax()) for i in range(len(proposals))]
        bonus = int(logits[-1].argmax())

        n_acc, nxt, full = accept_block(proposals, dense_choices, bonus)
        # dense-grade KV: pending + accepted proposals stay; rest cropped
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
        # acceptance accounting counts ONLY blocks fully inside the emitted
        # sequence: a final block cut by eos/max_new includes proposals a
        # sequential dense decode would never have needed, which inflates
        # acceptance on short-answer suites
        if consumed == len(emitted):
            stats["proposed"] += len(proposals)
            stats["accepted"] += n_acc
            stats["blocks"] += 1
            stats["full_blocks"] += int(full)
            stats["nacc"].append(n_acc)  # per-position acceptance: pos i
            # accepted iff n_acc > i (prefix acceptance is monotone), so the
            # nacc list reconstructs the full positional curve — including
            # the delta draft's structural pos-1 (dense-anchor) accept
            if not full:
                stats["first_reject_pos"].append(n_acc)
        pending = [generated[-1]] if generated[-1] not in eos else []
        if not pending:
            break

    return generated, stats


def dense_reference(model, tokenizer, prompt, max_new):
    """Plain dense-decode generation (the exactness target), via generate()."""
    model.config.decode_mode = "dense"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        t0 = time.monotonic()
        out = model.generate(**{k: v.cuda() for k, v in inputs.items()},
                             max_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()  # generate() can return with queued work
        dt = time.monotonic() - t0
    return out[0, inputs["input_ids"].size(1):].tolist(), dt


def dense_lean(model, tokenizer, prompt, max_new):
    """Sequential dense decode with the SAME lean machinery as spec_generate
    (pipeline prefill + q_len=1 rectangle steps + argmax). This is the fair
    wall-clock baseline: generate()'s per-step machinery (input_ids concat
    at 32K length, stopping criteria, logits processors) is harness tax a
    real serving loop doesn't pay, and charging it only to the dense side
    inflates speedups (07-16 review). Returns (tokens, seconds incl prefill)."""
    setattr(model, "no_lm_head", True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False)["input_ids"][0].tolist()
    _reset_dec_state(model)
    model.config._attn_implementation = model.config.attn_implementation_original
    t0 = time.monotonic()
    with torch.no_grad():
        out = model(torch.tensor([ids], device="cuda"), use_cache=True)
        logits = model.lm_head(out.logits[:, -1, :]).float()[0]
    model.config._attn_implementation = "sdpa_rectangle"
    model.config.decode_mode = "dense"
    cache = out.past_key_values
    generated = [int(logits.argmax())]
    while len(generated) < max_new and generated[-1] not in eos:
        lg, cache = _forward_tokens(model, [generated[-1]], cache)
        generated.append(int(lg[-1].argmax()))
    torch.cuda.synchronize()
    return generated, time.monotonic() - t0


def load_prompts(suite, tokenizer, n):
    """Returns (prompts, budgets) with chat templating ALREADY applied."""
    if suite == "govreport":
        rows = load_v1("gov_report")[:n]
        prompts = [truncate_middle(GOVREPORT_TEMPLATE.replace("{context}", r["context"]),
                                   tokenizer, MAX_PROMPT_TOKENS) for r in rows]
        return ([chat_wrap(p, tokenizer) for p in prompts],
                [GOVREPORT_MAX_NEW] * len(prompts))
    if suite == "qa":
        prompts, budgets = [], []
        per = max(n // len(V1_TASKS), 1)
        for task, (template, mx) in V1_TASKS.items():
            for ex in load_v1(task)[:per]:
                p = template.format(context=ex["context"], input=ex["input"])
                prompts.append(chat_wrap(
                    truncate_middle(p, tokenizer, MAX_PROMPT_TOKENS), tokenizer))
                budgets.append(mx)  # official per-task budget (32/32/32/64)
        return prompts, budgets
    if suite == "ruler":
        # negative control: needle retrieval, where the needle is outside the
        # draft's sink+window — the sparse draft cannot know the answer and
        # acceptance should collapse once positions leave local context.
        # prepare_task_data returns RULER's own llama-3-templated text, so no
        # chat_wrap here (double-templating would corrupt the prompt).
        from eval.ruler_client import prepare_task_data
        rows = prepare_task_data("niah_single_1", 32768, n, seed=0)
        return [r["input"] for r in rows], [64] * len(rows)
    raise SystemExit(f"unknown suite {suite}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["govreport", "qa", "ruler"], required=True)
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
    p.add_argument("--warm-baseline", action="store_true",
                   help="fair timing mode: after an untimed warmup, time a "
                        "LEAN sequential dense loop over every prompt as the "
                        "baseline, and time spec on every prompt too (07-16 "
                        "review: cold first-config generate() refs biased "
                        "tok_per_s_dense ~5% low)")
    p.add_argument("--out", type=str, default="results/specdec.csv")
    p.add_argument("--samples-out", type=str, default="results/specdec_samples.csv",
                   help="per-sample raw log (one row per config x prompt)")
    args = p.parse_args()

    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"specdec_{args.suite}", config=vars(args))

    def open_csv(path, cols):
        """Append-mode CSV with a schema guard: appending new-schema rows
        under an old header silently misaligns column-name-based analysis
        (07-16 review) — refuse instead."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with path.open() as f:
                first = f.readline().strip()
            if first and first.split(",") != cols:
                raise SystemExit(
                    f"{path} has a different column schema (header: "
                    f"{first[:100]}); appending would silently misalign "
                    "columns — pass a fresh --out/--samples-out")
            fh = path.open("a", newline="")
            return fh, csv.writer(fh)
        fh = path.open("a", newline="")
        wtr = csv.writer(fh)
        wtr.writerow(cols)
        return fh, wtr

    fh, w = open_csv(Path(args.out),
                     ["suite", "weights", "draft", "block", "n", "acceptance",
                      "acc_genuine", "pos_acc", "acc_per_verify",
                      "full_block_rate", "tok_per_s_spec", "tok_per_s_dense",
                      "parity_prefix_min", "run_id"])
    sfh, sw = open_csv(Path(args.samples_out),
                       ["suite", "weights", "draft", "block", "idx", "proposed",
                        "accepted", "blocks", "full_blocks", "nacc_hist",
                        "n_tokens", "t_prefill", "t_draft", "t_verify",
                        "run_id"])

    drafts = [d.strip() for d in args.drafts.split(",")]
    blocks = [int(b) for b in args.blocks.split(",")]
    weights = [x.strip() for x in args.weights.split(",")]
    bad = [d for d in drafts if d not in DRAFT_MODES] + \
          [x for x in weights if x not in DRAFT_WEIGHTS]
    if bad:
        raise SystemExit(f"unknown drafts/weights {bad}")

    for wkey in weights:
        model, tokenizer = build(wkey, args.gamma, args.window)
        prompts, budgets = load_prompts(args.suite, tokenizer, args.n_samples)
        # dense references are draft/block-independent: compute ONCE per
        # weights config (was recomputed per config — 6x waste on the grid).
        # Warmup first: the first CUDA forwards after model load pay kernel
        # autotune/pool costs, and charging that to t_dense would inflate the
        # reported speedup.
        if prompts:
            dense_reference(model, tokenizer, prompts[0], 16)
        refs = {}
        for i in range(min(args.exact_check_n, len(prompts))):
            refs[i] = dense_reference(model, tokenizer, prompts[i], budgets[i])
        lean_tok, lean_t = 0, 0.0
        if args.warm_baseline and prompts:
            # warm-vs-warm fair timing: one untimed spec + lean-dense pass
            # first (kernel autotune/pool costs), then the lean dense loop
            # over EVERY prompt as the baseline. The generate()-based refs
            # above remain the PARITY reference only — their timing is
            # neither lean nor warm (07-16 review) and is not used here.
            spec_generate(model, tokenizer, prompts[0], drafts[0], blocks[0],
                          budgets[0])
            dense_lean(model, tokenizer, prompts[0], budgets[0])
            for i, pr in enumerate(prompts):
                toks_l, dt_l = dense_lean(model, tokenizer, pr, budgets[i])
                lean_tok += len(toks_l)
                lean_t += dt_l
            print(f"[specdec] {wkey} lean warm dense baseline: "
                  f"{lean_tok / max(lean_t, 1e-9):.2f} tok/s "
                  f"({len(prompts)} prompts)", flush=True)
        for draft in drafts:
            for block in blocks:
                agg = {"proposed": 0, "accepted": 0, "verify_calls": 0,
                       "full_blocks": 0, "blocks": 0, "tokens": 0, "nacc": [],
                       # timing comparison uses ONLY the checked prompts and
                       # includes prefill on BOTH sides (dense_reference's
                       # generate() includes its prefill; spec adds t_prefill)
                       "tok_chk": 0, "t_spec_chk": 0.0,
                       "tok_ref": 0, "t_dense": 0.0, "exact": []}
                for i, prompt in enumerate(prompts):
                    toks, st = spec_generate(model, tokenizer, prompt, draft,
                                             block, budgets[i])
                    for k in ("proposed", "accepted", "verify_calls",
                              "full_blocks", "blocks"):
                        agg[k] += st[k]
                    agg["nacc"].extend(st["nacc"])
                    agg["tokens"] += len(toks)
                    hist = [st["nacc"].count(j) for j in range(block + 1)]
                    sw.writerow([args.suite, wkey, draft, block, i,
                                 st["proposed"], st["accepted"], st["blocks"],
                                 st["full_blocks"], "|".join(map(str, hist)),
                                 len(toks), f"{st['t_prefill']:.3f}",
                                 f"{st['t_draft']:.3f}",
                                 f"{st['t_verify']:.3f}", run.id])
                    sfh.flush()
                    if args.warm_baseline or i in refs:
                        # spec timing: every prompt under --warm-baseline,
                        # else just the checked ones
                        agg["tok_chk"] += len(toks)
                        agg["t_spec_chk"] += (st["t_prefill"] + st["t_draft"]
                                              + st["t_verify"])
                    if i in refs:
                        ref, dt = refs[i]
                        agg["tok_ref"] += len(ref)
                        agg["t_dense"] += dt
                        n_cmp = min(len(toks), len(ref))
                        div = next((j for j in range(n_cmp)
                                    if toks[j] != ref[j]), None)
                        eos_ids = model.generation_config.eos_token_id
                        eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple))
                                      else [eos_ids])
                        if div is not None:
                            parity_i = div
                            if args.min_parity_prefix and div < args.min_parity_prefix:
                                # early divergence: bug OR bf16 near-tie
                                # (GovReport prose hits ties as early as
                                # token 17) — one fresh dense forward
                                # separates the two before we hard-fail
                                margin = spec_token_margin(
                                    model, tokenizer, prompt, ref, div,
                                    toks[div])
                                print(f"[specdec] early divergence @{div}: "
                                      f"margin {margin:.3f}", flush=True)
                                if margin <= TIE_EPS:
                                    parity_i = ("tie", div, margin)
                        elif len(toks) != len(ref):
                            # clean prefix but different lengths is ALWAYS a
                            # harness bug (a legit eos stop matches through
                            # the eos, so lengths agree): hard-fail marker
                            parity_i = -1
                        elif toks and toks[-1] in eos_ids:
                            parity_i = 10**9  # byte-identical, natural eos
                        elif n_cmp >= max(args.min_parity_prefix, 1):
                            parity_i = 10**9  # byte-identical, long enough
                        else:
                            # byte-identical but SHORT with no eos ending —
                            # too few verified tokens to certify (e.g. a bug
                            # truncating both paths identically)
                            parity_i = n_cmp
                        agg["exact"].append(parity_i)
                    print(f"[specdec] {wkey}/{draft}/b{block} #{i}: "
                          f"acc {st['accepted']}/{st['proposed']}", flush=True)
                acc = agg["accepted"] / max(agg["proposed"], 1)
                # accepted per COUNTED verify: each counted block has exactly
                # one verify; final blocks cut by eos/max_new are excluded
                # from both sides (07-16 review: accepted/verify_calls mixed
                # populations — ~2x understated on short-budget suites)
                apv = agg["accepted"] / max(agg["blocks"], 1)
                fbr = agg["full_blocks"] / max(agg["blocks"], 1)
                pos_acc, genuine = positional_stats(agg["nacc"], block)
                tps_spec = agg["tok_chk"] / max(agg["t_spec_chk"], 1e-9)
                if args.warm_baseline:
                    tps_dense = lean_tok / max(lean_t, 1e-9)
                else:
                    tps_dense = agg["tok_ref"] / max(agg["t_dense"], 1e-9)
                parity, pfail = classify_parity(agg["exact"],
                                                args.min_parity_prefix)
                if pfail:
                    raise SystemExit(
                        f"PARITY FAILURE: {wkey}/{draft}/b{block} {pfail}; "
                        "aborting")
                w.writerow([args.suite, wkey, draft, block, len(prompts),
                            f"{acc:.4f}", f"{genuine:.4f}",
                            ";".join(f"{x:.4f}" for x in pos_acc),
                            f"{apv:.3f}", f"{fbr:.4f}",
                            f"{tps_spec:.2f}", f"{tps_dense:.2f}",
                            parity, run.id])
                fh.flush()
                run.log({f"specdec/{wkey}/{draft}/b{block}/acceptance": acc,
                         f"specdec/{wkey}/{draft}/b{block}/acc_genuine": genuine,
                         f"specdec/{wkey}/{draft}/b{block}/acc_per_verify": apv,
                         f"specdec/{wkey}/{draft}/b{block}/full_block_rate": fbr})
                run.summary[f"acc_{wkey}_{draft}_b{block}"] = acc
                run.summary[f"accg_{wkey}_{draft}_b{block}"] = genuine
                print(f"[specdec] {wkey}/{draft}/b{block}: acceptance {acc:.3f} "
                      f"genuine {genuine:.3f} "
                      f"pos_acc {';'.join(f'{x:.2f}' for x in pos_acc)} "
                      f"acc/verify {apv:.2f} full-block {fbr:.3f} "
                      f"parity_prefix_min={parity}", flush=True)
        model._sample = None
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    fh.close()
    sfh.close()
    run.finish()
    print("[specdec] DONE", flush=True)


if __name__ == "__main__":
    main()
