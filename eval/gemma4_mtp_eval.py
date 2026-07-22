"""Gemma 4 native MTP drafter — G0 baseline (docs/gemma4_mtp_plan.md).

STANDALONE by design: runs in .venv-g4 (transformers MAIN, which has
gemma4/gemma4_assistant) with NO delta_attention imports — our pinned
production stack never touches this process. Everything goes through the
vanilla HF API (`target.generate(assistant_model=assistant)`), because G0's
job is to certify the NATIVE pipeline before G1 reimplements the drafter.

Timing methodology (review wf_e370eeb8):
  - BOTH modes run on an explicit DynamicCache. transformers-main
    (5.15.0.dev0, probed 07-22 box 41) crashes assisted decoding when the
    prompt exceeds the trunk sliding window (1024): the hybrid cache
    window-caps sliding-layer KV while the verify mask is built full-length.
    DynamicCache sidesteps that; using it for the PLAIN baseline too keeps
    cache semantics symmetric (otherwise assisted pays full-length
    sliding-layer reads the baseline never pays).
  - Decode rate per mode comes from TWO generates (max_new and max_new/2):
    rate = (toks_long - toks_short) / (t_long - t_short). The difference
    cancels every one-time cost — target prefill, assistant setup/prompt
    pass, cache init — without modeling HF internals. (A max_new=1
    subtraction misses the assistant-side prompt pass and charges it to
    assisted decode only, deflating speedup as context grows.)
  - Parity: assisted greedy decoding is mathematically exact, so the
    assisted output must match plain greedy token-for-token up to kernel
    tie-flips. Gate: a row passes if exact OR its divergence point is
    >= --min-parity-prefix; the stage fails if fewer than half the rows
    pass (or none were measurable) — one late tie-flip cannot kill the
    run, a systematic wiring bug still does.
OOM at a tier is recorded as a row, not a crash.

    python eval/gemma4_mtp_eval.py --n 2 --tiers 4096 --max-new 64 \
        --min-parity-prefix 24 --out results/g4_smoke.csv
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

INSTR = ("Continue this story in the same style. Write a long, natural "
         "continuation.\n\n")

HEADER = ["tier", "idx", "prompt_toks",
          "plain_half_toks", "plain_half_s", "plain_toks", "plain_s",
          "plain_dec_tok_s",
          "asst_half_toks", "asst_half_s", "assisted_toks", "assisted_s",
          "assisted_dec_tok_s",
          "decode_speedup", "parity_prefix", "exact_match", "run_id"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10, help="prompts per tier")
    p.add_argument("--tiers", type=str, default="4096,16384,32768,65536",
                   help="max prompt tokens per tier")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--target", type=str, default=TARGET)
    p.add_argument("--assistant", type=str, default=ASSISTANT)
    p.add_argument("--min-parity-prefix", type=int, default=0,
                   help="per-row parity threshold (see module docstring); "
                        "0 disables the gate")
    p.add_argument("--out", type=str, default="results/g4_tiers.csv")
    return p.parse_args()


def load_contexts(tokenizer, n, max_ctx_tokens):
    """PG19 test books, token-truncated; deterministic dataset order. The
    instruction wrapper is applied later via the chat template."""
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    out = []
    for doc in ds:
        ids = tokenizer.encode(doc["text"], add_special_tokens=False)
        if len(ids) < max_ctx_tokens // 2:
            continue  # book must actually fill the tier
        out.append(tokenizer.decode(ids[:max_ctx_tokens],
                                    skip_special_tokens=True))
        if len(out) == n:
            return out
    raise SystemExit(f"only {len(out)} PG19 docs long enough for tier "
                     f"{max_ctx_tokens}")


@torch.no_grad()
def gen(model, inputs, max_new, assistant=None):
    """(new_token_ids, wall_seconds). Greedy; assisted iff assistant given.
    Always a fresh DynamicCache (see module docstring)."""
    from transformers import DynamicCache

    kw = dict(max_new_tokens=max_new, do_sample=False,
              past_key_values=DynamicCache())
    if assistant is not None:
        kw["assistant_model"] = assistant
    torch.cuda.synchronize()
    t0 = time.monotonic()
    ids = model.generate(**inputs, **kw)
    torch.cuda.synchronize()
    return ids[0, inputs["input_ids"].shape[1]:], time.monotonic() - t0


def parity_prefix(a: torch.Tensor, b: torch.Tensor) -> int:
    m = min(a.numel(), b.numel())
    neq = (a[:m] != b[:m]).nonzero()
    return int(neq[0].item()) if neq.numel() else m


def dec_rate(toks_short, t_short, toks_long, t_long):
    """Decode tok/s from the two-length difference; None when the deltas
    are too small to divide meaningfully (early eos, timer noise)."""
    dtok = toks_long - toks_short
    dt = t_long - t_short
    if dtok < 8 or dt < 0.05:
        return None
    return dtok / dt


def main():
    args = parse_args()
    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name="g4_mtp_g0", config=vars(args))

    import transformers
    # transformers is installed from git main (gemma4 support) — record the
    # exact build or rows are unattributable across reruns
    run.config.update({"transformers_version": transformers.__version__},
                      allow_val_change=True)
    print(f"[g4] transformers {transformers.__version__}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.target)
    print(f"[g4] loading target {args.target} (bf16)", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, device_map="cuda")
    print(f"[g4] loading assistant {args.assistant}", flush=True)
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=torch.bfloat16, device_map="cuda")
    target.eval()
    assistant.eval()
    nat = getattr(target.generation_config, "num_assistant_tokens", None)
    print(f"[g4] generation_config.num_assistant_tokens={nat}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():  # appending 17-col rows under an older header
        with out_path.open() as f:  # silently misaligns every column
            first = f.readline().strip().split(",")
        if first != HEADER:
            raise SystemExit(f"CSV schema mismatch in {out_path} — move or "
                             "delete the old file (or pass a fresh --out)")
    new = not out_path.exists()
    fh = out_path.open("a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(HEADER)

    def row(tier, idx, prompt_toks, **f):
        vals = [tier, idx, prompt_toks] + \
            [f.get(k, "") for k in HEADER[3:-1]] + [run.id]
        w.writerow(vals)
        fh.flush()

    half = max(args.max_new // 2, 8)
    gate_rows = []  # bool per measured row
    for tier in [int(x) for x in args.tiers.split(",")]:
        ctxs = load_contexts(tok, args.n, tier)
        rows_done = 0
        warmed = False  # retried until it succeeds — a prompt-0 OOM must
        # not leave the rest of the tier timed cold
        for i, ctx in enumerate(ctxs):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": INSTR + ctx}],
                add_generation_prompt=True, tokenize=False)
            inputs = tok(prompt, return_tensors="pt",
                         add_special_tokens=False).to("cuda")
            n_prompt = inputs["input_ids"].shape[1]
            try:
                if not warmed:  # compile/cache paths, both modes
                    gen(target, inputs, 16)
                    gen(target, inputs, 16, assistant)
                    warmed = True
                plain_h, t_ph = gen(target, inputs, half)
                plain, t_p = gen(target, inputs, args.max_new)
                asst_h, t_ah = gen(target, inputs, half, assistant)
                assisted, t_a = gen(target, inputs, args.max_new, assistant)
            except torch.cuda.OutOfMemoryError:
                row(tier, i, n_prompt, plain_toks="OOM")
                print(f"[g4] tier {tier} #{i}: OOM (recorded)", flush=True)
                torch.cuda.empty_cache()
                continue
            pp = parity_prefix(plain, assisted)
            # exact requires SAME length AND full match — a matching prefix
            # with different stopping points is a real divergence
            exact = int(plain.numel() == assisted.numel()
                        and pp == plain.numel())
            dp = dec_rate(plain_h.numel(), t_ph, plain.numel(), t_p)
            da = dec_rate(asst_h.numel(), t_ah, assisted.numel(), t_a)
            sp = (da / dp) if (dp and da) else None
            row(tier, i, n_prompt,
                plain_half_toks=plain_h.numel(), plain_half_s=f"{t_ph:.2f}",
                plain_toks=plain.numel(), plain_s=f"{t_p:.2f}",
                plain_dec_tok_s=f"{dp:.1f}" if dp else "",
                asst_half_toks=asst_h.numel(), asst_half_s=f"{t_ah:.2f}",
                assisted_toks=assisted.numel(), assisted_s=f"{t_a:.2f}",
                assisted_dec_tok_s=f"{da:.1f}" if da else "",
                decode_speedup=f"{sp:.3f}" if sp else "",
                parity_prefix=pp, exact_match=exact)
            if args.min_parity_prefix:
                gate_rows.append(bool(exact)
                                 or pp >= args.min_parity_prefix)
            rows_done += 1
            print(f"[g4] tier {tier} #{i}: decode plain "
                  f"{dp:.1f} tok/s" if dp else
                  f"[g4] tier {tier} #{i}: decode plain n/a", flush=True)
            print(f"      assisted {da:.1f} tok/s  speedup {sp:.2f}x"
                  if (da and sp) else "      assisted n/a", flush=True)
            print(f"      parity_prefix {pp}  exact={exact}  "
                  f"(walls p {t_ph:.1f}/{t_p:.1f}s a {t_ah:.1f}/{t_a:.1f}s)",
                  flush=True)
        run.summary[f"tier{tier}_rows"] = rows_done
    fh.close()

    if args.min_parity_prefix:
        if not gate_rows:
            raise SystemExit(
                "G0 GATE FAILED: no parity measurements at all (every row "
                "OOM'd) — the gate cannot certify an unmeasured harness")
        frac = sum(gate_rows) / len(gate_rows)
        run.summary["parity_pass_frac"] = frac
        if frac < 0.5:
            raise SystemExit(
                f"G0 GATE FAILED: only {sum(gate_rows)}/{len(gate_rows)} "
                "rows parity-pass — assisted greedy diverges from plain "
                "greedy systematically; harness or upstream bug")
        print(f"[g4] G0 gate PASS ({sum(gate_rows)}/{len(gate_rows)} rows)",
              flush=True)
    run.finish()
    print("[g4] DONE", flush=True)


if __name__ == "__main__":
    main()
