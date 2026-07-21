"""anchorbench (07-21): component-level decomposition of delta_forward_train
plus backend adjudication for the anchor branch.

Motivated by the swabench result (sparse-kernel choice moves step time
<=2.8%) and the critique that residual-based inference cannot localize the
overhead: this bench times each of the three pieces DIRECTLY on realistic
tensors — no full model, no checkpointing confounds — forward and
forward+backward, CUDA-synced, plus SDPA backend force-tests for the
anchor branch's masked call.

Variants (per the reviewed design):
  sparse-flex        compiled flex, block mask (the production sparse piece)
  gqa-expand         repeat_interleave of k/v alone (layout-copy cost)
  anchor-masked      current masked SDPA (baseline under test)
  anchor-flash!      sdpa_kernel(FLASH_ATTENTION) forced — expected to
                     RAISE if the mask blocks flash: the smoking gun,
                     logged either way
  anchor-mathonly    sdpa_kernel(MATH) forced — floor
  anchor-efficient   sdpa_kernel(EFFICIENT_ATTENTION) forced
  anchor-flexrow     flex with row-restricted mask over GATHERED anchor
                     queries (same k/v, prefixes in the mask — the
                     duplication-free reformulation candidate)
  correction         delta subtract + broadcast add + concat alone

Shapes mirror Llama-3.1-8B @ 32K training: q [1,32,S,128], kv [1,8,S,128]
bf16, gamma=64, window 2048, sink 1024. Single layer; multiply by 32 for
per-step estimates. Runs in ~2 minutes.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from delta_attention.train.flex_delta import (  # noqa: E402
    Q_BLOCK, _get_flex, anchor_layout, get_block_mask,
)
from torch.nn.attention.flex_attention import create_block_mask  # noqa: E402


def timed(fn, warmup=5, iters=20, backward=False):
    """(fwd_ms, fwdbwd_ms_or_None). Fresh graph per iter when backward."""
    for _ in range(warmup):
        out = fn()
        if backward:
            out.float().sum().backward()
    torch.cuda.synchronize()
    t0 = time.monotonic()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    fwd = (time.monotonic() - t0) / iters * 1000
    fb = None
    if backward:
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for _ in range(iters):
            out = fn()
            out.float().sum().backward()
        torch.cuda.synchronize()
        fb = (time.monotonic() - t0) / iters * 1000
    return fwd, fb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--gamma", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--sink", type=int, default=1024)
    p.add_argument("--out", type=str, default="results/anchorbench.csv")
    args = p.parse_args()
    s, gamma, window, sink = args.seq_len, args.gamma, args.window, args.sink
    assert s % Q_BLOCK == 0

    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"anchorbench_{s}", config=vars(args))

    dev, dt = "cuda", torch.bfloat16
    q = torch.randn(1, 32, s, 128, device=dev, dtype=dt, requires_grad=True)
    k8 = torch.randn(1, 8, s, 128, device=dev, dtype=dt, requires_grad=True)
    v8 = torch.randn(1, 8, s, 128, device=dev, dtype=dt, requires_grad=True)
    scaling = 128 ** -0.5

    idx, tail, s_p = anchor_layout(s, gamma)
    sel = torch.cat((idx, tail)).to(dev)
    n_sel = sel.numel()
    key_pos = torch.arange(s, device=dev)
    row_mask = (key_pos.unsqueeze(0) <= sel.unsqueeze(1)).view(1, 1, n_sel, s)
    print(f"[anchorbench] s={s} anchors+tail={n_sel} "
          f"(score tensor if materialized: "
          f"{32 * n_sel * s * 2 / 2**30:.2f} GB bf16/layer)", flush=True)

    def expand():
        return (k8.repeat_interleave(4, dim=1),
                v8.repeat_interleave(4, dim=1))

    k32, v32 = expand()
    k32r, v32r = k32.detach().requires_grad_(True), \
        v32.detach().requires_grad_(True)
    bm = get_block_mask(s, window, sink, dev)
    qs = q.detach()[:, :, sel].requires_grad_(True)  # gathered anchor queries

    sel_cpu = sel  # captured for the row-restricted flex mask
    def flexrow_mask(b, h, q_idx, kv_idx):
        return kv_idx <= sel_cpu[q_idx]
    bm_row = create_block_mask(flexrow_mask, B=None, H=None,
                               Q_LEN=n_sel, KV_LEN=s, device=dev)

    rows = []

    def record(label, fn, backward=True, note=""):
        try:
            fwd, fb = timed(fn, backward=backward)
            rows.append([label, s, f"{fwd:.2f}",
                         f"{fb:.2f}" if fb else "", note])
            print(f"[anchorbench] {label:18s} fwd {fwd:7.2f}ms  "
                  f"fwd+bwd {fb:7.2f}ms  {note}" if fb else
                  f"[anchorbench] {label:18s} fwd {fwd:7.2f}ms  {note}",
                  flush=True)
        except Exception as e:
            msg = str(e).splitlines()[0][:120]
            rows.append([label, s, "ERROR", "", msg])
            print(f"[anchorbench] {label:18s} RAISED: {msg}", flush=True)

    record("sparse-flex",
           lambda: _get_flex()(q, k32r, v32r, block_mask=bm, scale=scaling))
    record("gqa-expand", lambda: expand()[0], backward=False)
    record("anchor-masked",
           lambda: F.scaled_dot_product_attention(
               q[:, :, sel], k32r, v32r, attn_mask=row_mask, scale=scaling))

    def forced(backend):
        def fn():
            with sdpa_kernel(backends=[backend]):
                return F.scaled_dot_product_attention(
                    q[:, :, sel], k32r, v32r, attn_mask=row_mask,
                    scale=scaling)
        return fn
    record("anchor-flash!", forced(SDPBackend.FLASH_ATTENTION),
           note="raises => mask blocks flash (the smoking gun)")
    record("anchor-mathonly", forced(SDPBackend.MATH))
    record("anchor-efficient", forced(SDPBackend.EFFICIENT_ATTENTION))
    record("anchor-flexrow",
           lambda: _get_flex()(qs, k32r, v32r, block_mask=bm_row,
                               scale=scaling),
           note="gathered rows, same k/v, prefixes via mask (no duplication)")

    sparse_out = _get_flex()(q, k32, v32, block_mask=bm,
                             scale=scaling).transpose(1, 2).detach() \
        .requires_grad_(True)
    anchor_out = torch.randn(1, n_sel, 32, 128, device=dev, dtype=dt,
                             requires_grad=True)

    def correction():
        delta = anchor_out[:, :idx.numel()] - sparse_out[:, idx.to(dev)]
        n = idx.numel()
        corrected = (sparse_out[:, :s_p].reshape(1, n, gamma, 32, 128)
                     + delta.reshape(1, n, 1, 32, 128)).reshape(1, s_p, 32, 128)
        return torch.cat((corrected, anchor_out[:, idx.numel():]), dim=1)
    record("correction", correction)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if fh.tell() == 0:
            w.writerow(["component", "seq_len", "fwd_ms", "fwdbwd_ms", "note"])
        w.writerows(rows)
    for r in rows:
        if r[2] != "ERROR":
            run.summary[f"{r[0]}_fwd_ms"] = float(r[2])
    run.finish()
    print("[anchorbench] DONE", flush=True)


if __name__ == "__main__":
    main()
