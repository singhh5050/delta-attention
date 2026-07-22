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
    full     — cropped, untouched (the certification arm: final tokens
               must equal native generate(assistant_model=...) exactly)
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
    p.add_argument("--out", type=str, default="results/g1_tiers.csv")
    return p.parse_args()


def load_contexts(tokenizer, n, max_ctx_tokens):
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    out = []
    for doc in ds:
        ids = tokenizer.encode(doc["text"], add_special_tokens=False)
        if len(ids) < max_ctx_tokens // 2:
            continue
        out.append(tokenizer.decode(ids[:max_ctx_tokens],
                                    skip_special_tokens=True))
        if len(out) == n:
            return out
    raise SystemExit(f"only {len(out)} PG19 docs long enough for tier "
                     f"{max_ctx_tokens}")


def crop_kv(kv, length):
    return {k: (v[0][:, :, :length, :], v[1][:, :, :length, :])
            for k, v in kv.items()}


def arm_view(kv, arm, call_idx):
    """The intervention: how THIS drafter call sees the trunk KV.
    kv is already cropped to the validated prefix."""
    if arm == "full":
        return kv
    if arm.startswith("delta"):
        n = int(arm[len("delta"):])
        if call_idx % n == 0:  # anchor-refresh call reads everything
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


_LAST_STAGE = "?"  # which GPU call OOM'd (module-global; single-threaded)
_OOM_DUMPED = False


def _st(s):
    global _LAST_STAGE
    _LAST_STAGE = s


@torch.no_grad()
def run_arm(target, assistant, embed, ids, arm, k_drafts, max_new):
    """Our draft-verify loop. Returns per-prompt stats dict.

    Cache invariant: the trunk DynamicCache holds the validated prefix
    EXCLUDING the newest validated token; each verify forward feeds
    [newest_validated, d1..dK], so logits[i] checks draft i+1 and
    hidden[j] is the hidden of the last validated token when j drafts
    match (mirrors the native generator's n_last_matches indexing).
    """
    from transformers import DynamicCache

    dev = ids.device
    cache = DynamicCache()
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
    kv = out.shared_kv_states
    validated = ids[0].tolist()
    newest = None  # newest validated token, not yet in trunk cache

    # first token is the trunk's own greedy choice
    t0 = int(pending_logit.argmax(-1).item())
    validated.append(t0)
    newest = t0

    rounds = drafted = accepted = bonus_ct = 0
    call_ms = []
    while len(validated) - ids.shape[1] < max_new:
        L = len(validated)
        # NATIVE-EXACT crop: the generator slices KV to current_length =
        # len(validated INCLUDING the bonus token), even though the trunk
        # never forwarded the bonus — so the row at the bonus position is
        # the REJECTED draft's KV (or absent when all drafts matched:
        # slicing past the end just returns what exists). Deliberately
        # replicated, staleness and all, so the full arm is bit-comparable
        # to native assisted decoding (the parity gate depends on it).
        kv_c = crop_kv(kv, L)
        last_tok = torch.tensor([[newest]], device=dev)
        pos = torch.tensor([[L - 1]], dtype=torch.long, device=dev)
        drafts = []
        h = last_hidden
        t = last_tok
        for c in range(k_drafts):
            emb = embed(t)
            inp = torch.cat([emb, h], dim=-1)
            _st("draft_call")
            torch.cuda.synchronize()
            tt0 = time.monotonic()
            o = assistant(inputs_embeds=inp, attention_mask=None,
                          position_ids=pos,
                          shared_kv_states=arm_view(kv_c, arm, c),
                          use_cache=False)
            torch.cuda.synchronize()
            call_ms.append((time.monotonic() - tt0) * 1000)
            t = o.logits.argmax(dim=-1)
            h = o.last_hidden_state
            drafts.append(int(t.item()))
        rounds += 1
        drafted += len(drafts)

        # trunk verifies [newest, d1..dK]
        _st("verify")
        vin = torch.tensor([[newest] + drafts], device=dev)
        vout = target(input_ids=vin, past_key_values=cache, use_cache=True,
                      output_hidden_states=True,
                      return_shared_kv_states=True)
        vlogits = vout.logits[0]  # (K+1, vocab)
        n_match = 0
        for j, d in enumerate(drafts):
            if int(vlogits[j].argmax(-1).item()) == d:
                n_match += 1
            else:
                break
        accepted += n_match
        b = int(vlogits[n_match].argmax(-1).item())
        bonus_ct += 1
        validated.extend(drafts[:n_match] + [b])
        newest = b
        last_hidden = vout.hidden_states[-1][:, n_match:n_match + 1]
        kv = vout.shared_kv_states
        # cache now holds old + K+1 tokens; keep only old + newest_prev +
        # accepted drafts (the invariant: exclude the NEW newest = bonus)
        cache.crop(len(validated) - 1)

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
                                args.k, args.max_new)
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
                    torch.cuda.empty_cache()
                    continue
                if arm == "full":
                    ref_tokens = r["tokens"]
                match = ("" if ref_tokens is None or arm == "full"
                         else int(r["tokens"] == ref_tokens))
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
                from transformers import DynamicCache
                with torch.no_grad():
                    nat = target.generate(
                        input_ids=ids, max_new_tokens=args.max_new,
                        do_sample=False, assistant_model=assistant,
                        past_key_values=DynamicCache())
                nat_new = nat[0, ids.shape[1]:].tolist()
                m = 0
                for a, b in zip(nat_new, ref_tokens or []):
                    if a != b:
                        break
                    m += 1
                print(f"[g1] parity vs native: prefix {m}/"
                      f"{min(len(nat_new), len(ref_tokens or []))}",
                      flush=True)
                run.summary[f"parity_prefix_t{tier}_p{i}"] = m
                if m < min(24, args.max_new // 2):
                    raise SystemExit(
                        "G1 GATE FAILED: our loop diverges from native "
                        f"assisted decoding at token {m} — loop bug")
    fh.close()
    run.finish()
    print("[g1] DONE", flush=True)


if __name__ == "__main__":
    main()
