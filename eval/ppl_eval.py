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
    p.add_argument("--forward", choices=["pipeline", "dense"], default="pipeline",
                   help="eval-time attention: the delta pipeline forward or plain "
                        "dense (the missing 2x2 cell: does delta-training help "
                        "only under the pipeline, or generically?)")
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--gamma", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--adapters-root", type=str, default="checkpoints")
    p.add_argument("--data-source", choices=["pg19", "arxiv"], default="pg19",
                   help="held-out corpus for the chunks (arxiv = T3 replication)")
    return p.parse_args()


def test_chunks(tokenizer, seq_len, n, source="pg19"):
    from datasets import load_dataset

    if source == "arxiv":
        # T3 second-corpus eval. common-pile/arxiv_papers has ONE split;
        # training streams SHUFFLED from the head (~1.5K docs consumed at
        # 500x32K), so held-out chunks come from 20K docs deep — disjoint
        # by construction. No shuffle here: deterministic chunks are what
        # make the per-chunk stats pairable across arms. Cross-doc packed
        # (papers < 32K tokens), eos-separated, mirroring the training
        # packing.
        ds = load_dataset("common-pile/arxiv_papers", split="train",
                          streaming=True).skip(20000)
        eos = tokenizer.eos_token_id
        buf, out = [], []
        for doc in ds:
            text = doc.get("text") or ""
            if len(text) < 2000:
                continue
            buf.extend(tokenizer.encode(text, add_special_tokens=False))
            buf.append(eos)
            while len(buf) >= seq_len:
                out.append(torch.tensor([buf[:seq_len]]))
                buf = buf[seq_len:]
                if len(out) == n:
                    return out
        raise SystemExit(f"only {len(out)} arxiv test chunks available")

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
    return mean, math.exp(mean), losses


def main():
    args = parse_args()
    import wandb

    from delta_attention.config import Config
    from delta_attention.sample import init_model

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"wp2_posttrain_ppl_{args.forward}", config=vars(args))
    results = {}
    tokenizer_ref = None
    chunks = None

    for arm in args.arms.split(","):
        cfg = Config()
        if args.forward == "dense":
            cfg.attn_implementation = "flash_attention_2"
            cfg.mode = "none"
        else:
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
        model.config.log_drift = False
        model.config.detach_delta = False
        if args.forward == "pipeline":
            model.config._attn_implementation = "flex_delta_train"  # pipeline forward, no-grad
        model.config.use_cache = False
        model.eval().cuda()

        if chunks is None:
            chunks = test_chunks(tokenizer, args.seq_len, args.chunks,
                                 source=args.data_source)
        loss, ppl, per_chunk = pipeline_ppl(model, chunks)
        results[arm] = (loss, ppl, per_chunk)
        print(f"[ppl] {arm}: loss {loss:.4f}  ppl {ppl:.3f}  "
              f"({args.chunks} held-out {args.data_source} chunks @ {args.seq_len})", flush=True)
        print(f"[ppl] {arm} per-chunk: {[round(x, 4) for x in per_chunk]}", flush=True)
        run.log({f"ppl/{arm}": ppl, f"loss/{arm}": loss})
        run.summary[f"per_chunk_loss_{arm}"] = per_chunk
        # the _sample monkey-patch creates a self-reference cycle, so refcount
        # freeing never fires — collect explicitly or the next arm OOMs on 40GB
        model._sample = None
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    run.summary.update({f"ppl_{k}": v[1] for k, v in results.items()})
    # paired per-chunk deltas: identical chunks across arms, so mean±std of
    # the difference is the sensitive test for small gaps between arms
    arms = list(results)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            d = [x - y for x, y in zip(results[a][2], results[b][2])]
            m = sum(d) / len(d)
            sd = (sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)) ** 0.5
            run.summary[f"paired_{a}_minus_{b}"] = m
            print(f"[ppl] paired loss {a}-{b}: {m:+.4f} ± {sd / len(d) ** 0.5:.4f} (sem)",
                  flush=True)
    run.finish()
    print("[ppl] DONE:", {k: round(v[1], 3) for k, v in results.items()}, flush=True)


if __name__ == "__main__":
    main()
