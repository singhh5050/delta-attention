"""Sparse-branch kernel diagnostic (Jeff, 07-21): is the flex block mask the
overhead, the GQA materialization, or the anchor branch?

One model load per seq-len; variants loop IN-PROCESS (the weights are
identical — only the attention dispatch changes), each with its own warmup.
CUDA-synced fwd/bwd timing over full training steps, same discipline as
trainbench (O4). fa2swa rows are TIMING-ONLY (no sink -> knowingly wrong
math; gamma=1 identity still holds so nothing explodes).

    python eval/swa_bench.py --seq-len 32768 --steps 30
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from delta_attention.train.train_delta import (  # noqa: E402
    build_model, chunked_ce_hidden, packed_stream,
)

# (label, model _attn_implementation, DELTA_SPARSE_IMPL env)
VARIANTS = [
    ("delta-flex", "flex_delta_train", "flex"),          # production baseline
    ("delta-flexgqa", "flex_delta_train", "flexgqa"),    # native-GQA flex
    ("delta-fa2swa2048", "flex_delta_train", "fa2swa"),  # Jeff's diagnostic
    ("delta-fa2swa3072", "flex_delta_train", "fa2swa"),  # +sink-sized bracket
    ("dense-fa2", "flash_attention_2", ""),              # dense reference
]


def gpu_state():
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return [x.strip() for x in q.split(",")[:3]]
    except Exception:
        return ["?", "?", "?"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--out", type=str, default="results/swabench.csv")
    args = p.parse_args()
    # each variant's warmup must absorb its OWN dynamo/flex recompile
    # (enable_gqa and kernel switches recompile mid-loop); 0 would fold
    # compile time into timed steps
    if args.warmup < 5:
        print(f"[swabench] WARNING: raising --warmup {args.warmup} -> 5 "
              "(recompile absorption floor)", flush=True)
        args.warmup = 5

    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"swabench_{args.seq_len}", config=vars(args))

    class A:  # the arg surface build_model consumes
        model = ""
        arm = "delta"
        gamma = 64
        window = 2048
        detach_delta = False
        delta_grad_scale = 1.0
        dense_impl = "flash_attention_2"

    model, tokenizer = build_model(A())
    lm_w = model.get_base_model().lm_head.weight
    data = packed_stream(tokenizer, args.seq_len, source="pg19", seed=0)
    # fixed batch set: identical inputs for every variant (content doesn't
    # change compute shape, but identical is free so why not)
    batches = [next(data).cuda() for _ in range(args.warmup + args.steps)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists()
    fh = out_path.open("a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["variant", "seq_len", "n_timed", "fwd_ms", "bwd_ms",
                    "step_ms", "tok_per_s", "sm_mhz", "sm_max_mhz",
                    "gpu_temp_c", "run_id"])

    opt = torch.optim.AdamW((q for q in model.parameters()
                             if q.requires_grad), lr=1e-5)
    for label, impl, env in VARIANTS:
        model.config._attn_implementation = impl
        if env:
            os.environ["DELTA_SPARSE_IMPL"] = env
        else:
            os.environ.pop("DELTA_SPARSE_IMPL", None)
        os.environ["DELTA_FA2_WINDOW"] = \
            "3072" if label.endswith("3072") else "2048"
        fwd, bwd, tot = [], [], []
        for i, batch in enumerate(batches):
            torch.cuda.synchronize()
            t0 = time.monotonic()
            out = model(input_ids=batch)
            loss = chunked_ce_hidden(out.logits, lm_w, batch)
            torch.cuda.synchronize()
            t1 = time.monotonic()
            loss.backward()
            torch.cuda.synchronize()
            t2 = time.monotonic()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t3 = time.monotonic()
            if i >= args.warmup:
                fwd.append(t1 - t0)
                bwd.append(t2 - t1)
                tot.append(t3 - t0)
        n = len(tot)
        sm, sm_max, temp = gpu_state()
        row = [label, args.seq_len, n,
               f"{1000*sum(fwd)/n:.1f}", f"{1000*sum(bwd)/n:.1f}",
               f"{1000*sum(tot)/n:.1f}",
               f"{args.seq_len*n/sum(tot):.0f}", sm, sm_max, temp, run.id]
        w.writerow(row)
        fh.flush()
        run.summary[f"step_ms_{label}_{args.seq_len}"] = 1000 * sum(tot) / n
        print(f"[swabench] {label} @{args.seq_len}: "
              f"fwd {1000*sum(fwd)/n:.0f}ms bwd {1000*sum(bwd)/n:.0f}ms "
              f"step {1000*sum(tot)/n:.0f}ms (clocks {sm}/{sm_max}MHz "
              f"{temp}C)", flush=True)
    fh.close()
    run.finish()
    print("[swabench] DONE", flush=True)


if __name__ == "__main__":
    main()
