"""Gemma 4 native MTP drafter — G0 baseline (docs/gemma4_mtp_plan.md).

STANDALONE by design: runs in .venv-g4 (transformers MAIN, which has
gemma4/gemma4_assistant) with NO delta_attention imports — our pinned
production stack never touches this process. Everything goes through the
vanilla HF API (`target.generate(assistant_model=assistant)`), because G0's
job is to certify the NATIVE pipeline before G1 reimplements the drafter.

Measures, per context tier:
  - plain greedy decode tok/s (target alone)
  - assisted greedy decode tok/s (target + jointly-trained drafter)
  - speedup, and OUTPUT PARITY between the two. Assisted greedy decoding is
    mathematically exact, so divergence beyond kernel-nondeterminism tie
    flips is a harness/upstream bug (same parity-gate discipline as
    specdec_eval). Parity here compares token prefixes and records the
    first-divergence index; the smoke gate requires a minimum prefix.
OOM at a tier is recorded as a row, not a crash (the 31B trunk + 262K KV
may not fit 80GB; that boundary is itself a G0 result).

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10, help="prompts per tier")
    p.add_argument("--tiers", type=str, default="4096,16384,32768,65536",
                   help="max prompt tokens per tier")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--target", type=str, default=TARGET)
    p.add_argument("--assistant", type=str, default=ASSISTANT)
    p.add_argument("--min-parity-prefix", type=int, default=0,
                   help="smoke gate: min tokens before first divergence "
                        "(mean); 0 disables")
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
    """(new_token_ids, wall_seconds). Greedy; assisted iff assistant given."""
    kw = dict(max_new_tokens=max_new, do_sample=False)
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


def main():
    args = parse_args()
    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name="g4_mtp_g0", config=vars(args))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.target)
    print(f"[g4] loading target {args.target} (bf16)", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map="cuda")
    print(f"[g4] loading assistant {args.assistant}", flush=True)
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, torch_dtype=torch.bfloat16, device_map="cuda")
    target.eval()
    assistant.eval()
    nat = getattr(target.generation_config, "num_assistant_tokens", None)
    print(f"[g4] generation_config.num_assistant_tokens={nat}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists()
    fh = out_path.open("a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["tier", "idx", "prompt_toks", "plain_toks", "plain_s",
                    "assisted_toks", "assisted_s", "speedup",
                    "parity_prefix", "exact_match", "run_id"])

    gate_vals = []
    for tier in [int(x) for x in args.tiers.split(",")]:
        ctxs = load_contexts(tok, args.n, tier)
        rows_done = 0
        for i, ctx in enumerate(ctxs):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": INSTR + ctx}],
                add_generation_prompt=True, tokenize=False)
            inputs = tok(prompt, return_tensors="pt",
                         add_special_tokens=False).to("cuda")
            n_prompt = inputs["input_ids"].shape[1]
            try:
                if i == 0:  # per-tier warmup: compile/cache paths, both modes
                    gen(target, inputs, 16)
                    gen(target, inputs, 16, assistant)
                plain, t_p = gen(target, inputs, args.max_new)
                assisted, t_a = gen(target, inputs, args.max_new, assistant)
            except torch.cuda.OutOfMemoryError:
                w.writerow([tier, i, n_prompt, "OOM", "", "", "", "", "", "",
                            run.id])
                fh.flush()
                print(f"[g4] tier {tier} #{i}: OOM (recorded)", flush=True)
                torch.cuda.empty_cache()
                continue
            if plain.numel() == 0 or assisted.numel() == 0:
                w.writerow([tier, i, n_prompt, plain.numel(), f"{t_p:.2f}",
                            assisted.numel(), f"{t_a:.2f}", "", "", "",
                            run.id])
                fh.flush()
                print(f"[g4] tier {tier} #{i}: empty generation (immediate "
                      "eos) — recorded, skipped for stats", flush=True)
                continue
            pp = parity_prefix(plain, assisted)
            exact = int(pp == min(plain.numel(), assisted.numel()))
            sp = (assisted.numel() / t_a) / (plain.numel() / t_p)
            w.writerow([tier, i, n_prompt, plain.numel(), f"{t_p:.2f}",
                        assisted.numel(), f"{t_a:.2f}", f"{sp:.3f}",
                        pp, exact, run.id])
            fh.flush()
            gate_vals.append(pp)
            rows_done += 1
            print(f"[g4] tier {tier} #{i}: plain {plain.numel()/t_p:.1f} tok/s"
                  f"  assisted {assisted.numel()/t_a:.1f} tok/s"
                  f"  speedup {sp:.2f}x  parity_prefix {pp}"
                  f"  exact={exact}", flush=True)
        run.summary[f"tier{tier}_rows"] = rows_done
    fh.close()

    if args.min_parity_prefix and gate_vals:
        mean_pp = sum(gate_vals) / len(gate_vals)
        run.summary["mean_parity_prefix"] = mean_pp
        if mean_pp < args.min_parity_prefix:
            raise SystemExit(
                f"G0 GATE FAILED: mean parity prefix {mean_pp:.1f} < "
                f"{args.min_parity_prefix} — assisted greedy diverges from "
                "plain greedy far too early; harness or upstream bug")
        print(f"[g4] G0 gate PASS (mean parity prefix {mean_pp:.1f})",
              flush=True)
    run.finish()
    print("[g4] DONE", flush=True)


if __name__ == "__main__":
    main()
