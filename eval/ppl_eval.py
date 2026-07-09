"""Held-out perplexity under the delta pipeline, trained vs base.

Answers "how did perplexity look after training" for the WP-2 pilot: computes
CE/perplexity on PG19 TEST-split chunks (disjoint from training) under the
differentiable pipeline forward (T13-equivalent to the inference path), for
the base model and each pilot adapter.

    python eval/ppl_eval.py --arms base,delta,dense,detach --chunks 16
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", type=str, default="base,delta,dense,detach")
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--gamma", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--adapters-root", type=str, default="checkpoints")
    return p.parse_args()


def test_chunks(tokenizer, seq_len, n):
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    out = []
    for doc in ds:
        toks = tokenizer.encode(doc["text"], add_special_tokens=False)
        for c in range(min(len(toks) // seq_len, 2)):  # ≤2 chunks/doc for diversity
            out.append(torch.tensor([toks[c * seq_len:(c + 1) * seq_len]]))
            if len(out) == n:
                return out
    raise SystemExit(f"only {len(out)} test chunks available")


@torch.no_grad()
def pipeline_ppl(model, chunks):
    losses = []
    for ids in chunks:
        out = model(input_ids=ids.cuda(), labels=ids.cuda(), use_cache=False)
        losses.append(out.loss.item())
    mean = sum(losses) / len(losses)
    return mean, math.exp(mean)


def main():
    args = parse_args()
    import wandb

    from delta_attention.config import Config
    from delta_attention.sample import init_model

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name="wp2_posttrain_ppl", config=vars(args))
    results = {}
    tokenizer_ref = None
    chunks = None

    for arm in args.arms.split(","):
        cfg = Config()
        cfg.attn_implementation = "window"
        cfg.mode = "delta"
        cfg.delta_lambda = args.gamma
        cfg.sliding_window = args.window
        cfg.attn_implementation_original = cfg.attn_implementation
        if arm != "base":
            path = Path(args.adapters_root) / f"pilot_{arm}"
            assert path.exists(), f"adapter missing: {path} (download from wandb first)"
            cfg.checkpoint = str(path)
        model, tokenizer = init_model(cfg)
        model.config.delta_lambda = args.gamma
        model.config.sliding_window = args.window
        model.config.log_drift = False
        model.config.detach_delta = False
        model.config._attn_implementation = "flex_delta_train"  # pipeline forward, no-grad
        model.config.use_cache = False
        model.eval().cuda()

        if chunks is None:
            chunks = test_chunks(tokenizer, args.seq_len, args.chunks)
        loss, ppl = pipeline_ppl(model, chunks)
        results[arm] = (loss, ppl)
        print(f"[ppl] {arm}: loss {loss:.4f}  ppl {ppl:.3f}  "
              f"({args.chunks} held-out PG19-test chunks @ {args.seq_len})", flush=True)
        run.log({f"ppl/{arm}": ppl, f"loss/{arm}": loss})
        del model
        torch.cuda.empty_cache()

    run.summary.update({f"ppl_{k}": v[1] for k, v in results.items()})
    run.finish()
    print("[ppl] DONE:", {k: round(v[1], 3) for k, v in results.items()}, flush=True)


if __name__ == "__main__":
    main()
