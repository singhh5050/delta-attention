"""anchorbench v2 (07-21, post-review wf_c5d06bb1): component-level
decomposition of delta_forward_train + SDPA backend adjudication + Jeff's
weightless long-context ladder (131K -> 1M) + MTP head-read cost curves.

ALL NUMBERS ARE PER ATTENTION LAYER, PER CALL (a 'scope' column says so;
multiply by n_layers for per-step estimates).

Methodology (each item exists because review wf_c5d06bb1 confirmed its
absence distorted v1):
- Symmetric work: every anchor cell gathers q[:, :, sel] IN-GRAPH inside
  the timed fn from the same full-q leaf, so forward gather + backward
  scatter into q.grad is paid identically by masked-SDPA and flexrow.
- Production-faithful GQA: cells that use expanded k/v perform
  repeat_interleave IN-GRAPH inside the timed fn (backward pays the
  32->8-head grad reduction exactly as production does); GQA-native cells
  consume the 8-head leaves directly.
- Output-size-independent backward: loss surrogate is out.backward(g)
  with a pre-allocated bf16 gradient — no fp32 upcast, no reduction.
- Grads zeroed between iterations (accumulation would grow allocator
  pressure across 20 iters).
- Failures PROPAGATE (stage fails loudly). Exactly two exceptions are
  data, not failure: the anchor-flash! cell (an error IS the smoking gun)
  and CUDA OOM on the long-context ladder (an OOM row documents that the
  formulation cannot run at that length — e.g. the masked-SDPA row mask
  alone is ~16.5GB at 1M).
- Per-cell tensor lifecycle + empty_cache: a 1M cell's transients must
  not degrade the next cell.

    python eval/anchor_bench.py --seq-lens 8192,32768
    python eval/anchor_bench.py --seq-lens 131072,262144,524288,1048576
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import subprocess
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

SINK, WINDOW, GAMMA = 1024, 2048, 64
H_Q, H_KV, D = 32, 8, 128  # Llama-3.1-8B geometry


def gpu_state():
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return [x.strip() for x in q.splitlines()[0].split(",")[:3]]
    except Exception:
        return ["?", "?", "?"]


def timed(fn, out_shape, warmup, iters, leaves, backward=True):
    """(fwd_ms, fwdbwd_ms|None). backward via out.backward(g) with a
    pre-allocated bf16 g (size-independent surrogate); leaves' grads
    zeroed each iteration."""
    g = torch.randn(out_shape, device="cuda", dtype=torch.bfloat16) \
        if backward else None

    def zero():
        for t in leaves:
            t.grad = None
    for _ in range(warmup):
        out = fn()
        if backward:
            out.backward(g)
            zero()
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
            out.backward(g)
            zero()
        torch.cuda.synchronize()
        fb = (time.monotonic() - t0) / iters * 1000
    return fwd, fb


def bench_seq_len(s, warmup, iters, rows, run):
    assert s % Q_BLOCK == 0
    # long-ladder cells cost seconds per iteration — scale counts so 1M
    # doesn't run for hours (Jeff's ask is the curve, not 20 replicates)
    if s >= 524288:
        warmup, iters = min(warmup, 3), min(iters, 3)
    elif s >= 131072:
        warmup, iters = min(warmup, 4), min(iters, 6)
    dev, dt = "cuda", torch.bfloat16
    q = torch.randn(1, H_Q, s, D, device=dev, dtype=dt, requires_grad=True)
    k8 = torch.randn(1, H_KV, s, D, device=dev, dtype=dt, requires_grad=True)
    v8 = torch.randn(1, H_KV, s, D, device=dev, dtype=dt, requires_grad=True)
    leaves = [q, k8, v8]
    scaling = D ** -0.5
    idx, tail, s_p = anchor_layout(s, GAMMA)
    sel = torch.cat((idx, tail)).to(dev)
    n_sel = sel.numel()
    rep = H_Q // H_KV
    print(f"[anchorbench] s={s}: anchors+tail={n_sel}; masked-SDPA row mask "
          f"= {n_sel * s / 8 / 2**30:.2f} GB bool; materialized scores would "
          f"be {H_Q * n_sel * s * 2 / 2**30:.2f} GB bf16", flush=True)

    def cell(label, fn, out_shape, backward=True, expect_raise=False,
             note=""):
        try:
            fwd, fb = timed(fn, out_shape, warmup, iters, leaves,
                            backward=backward)
            rows.append([label, s, "per-layer-fwd", f"{fwd:.2f}",
                         f"{fb:.2f}" if fb else "", note])
            run.summary[f"{label}_{s}_fwd_ms"] = fwd
            if fb:
                run.summary[f"{label}_{s}_fwdbwd_ms"] = fb
            print(f"[anchorbench] {label:18s} s={s:>7} PER-LAYER "
                  f"fwd {fwd:8.2f}ms" + (f"  fwd+bwd {fb:8.2f}ms" if fb
                                          else "") + f"  {note}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            rows.append([label, s, "per-layer-fwd", "OOM", "OOM",
                         "formulation cannot run at this length on 80GB"])
            print(f"[anchorbench] {label:18s} s={s:>7} OOM (recorded as "
                  "data)", flush=True)
            for t in leaves:
                t.grad = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            if expect_raise:
                msg = str(e).splitlines()[0][:120]
                rows.append([label, s, "per-layer-fwd", "RAISED", "", msg])
                print(f"[anchorbench] {label:18s} s={s:>7} RAISED: {msg} "
                      "(the smoking gun, recorded)", flush=True)
            elif "64-bit indexing" in str(e):
                # torch 2.8 flex triton templates are int32-indexed: any
                # tensor over 2^31 elements (q at s>=524288 with 32 heads)
                # cannot compile. A formulation cap, so a RESULT — the
                # -hc cells below carry the delta curve past it.
                rows.append([label, s, "per-layer-fwd", "RAISED-64BIT", "",
                             "flex triton templates lack 64-bit indexing "
                             "(torch 2.8): >2^31-element tensors"])
                print(f"[anchorbench] {label:18s} s={s:>7} RAISED-64BIT "
                      "(recorded as data)", flush=True)
                for t in leaves:
                    t.grad = None
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise  # genuine failures fail the stage loudly

    bm = get_block_mask(s, WINDOW, SINK, dev)

    # head-chunked flex: N sequential head-group calls, each under the
    # int32 element limit of flex's triton templates. GQA group alignment
    # holds for both expanded and native kv (q head i <-> kv head i//rep,
    # and chunks split both on the same group boundaries). Costs only N-1
    # extra kernel launches — the honest workaround to carry the curve to
    # 1M, labeled -hcN so nobody mistakes it for the monolithic kernel.
    HC = 4

    def flex_hc(qq, kk, vv, block_mask, enable_gqa=False):
        gq, gkv = qq.shape[1] // HC, kk.shape[1] // HC
        return torch.cat(
            [_get_flex()(qq[:, i * gq:(i + 1) * gq],
                         kk[:, i * gkv:(i + 1) * gkv],
                         vv[:, i * gkv:(i + 1) * gkv],
                         block_mask=block_mask, scale=scaling,
                         enable_gqa=enable_gqa)
             for i in range(HC)], dim=1)

    needs_hc = H_Q * s * D >= 2 ** 31  # monolithic flex will RAISED-64BIT

    # ---- production sparse branch: expansion IN-GRAPH (backward pays the
    # 32->8 grad reduction, exactly as delta_forward_train does)
    cell("sparse-flex",
         lambda: _get_flex()(q, k8.repeat_interleave(rep, dim=1),
                             v8.repeat_interleave(rep, dim=1),
                             block_mask=bm, scale=scaling),
         (1, H_Q, s, D))
    if needs_hc:
        cell(f"sparse-flex-hc{HC}",
             lambda: flex_hc(q, k8.repeat_interleave(rep, dim=1),
                             v8.repeat_interleave(rep, dim=1), bm),
             (1, H_Q, s, D), note=f"{HC} head-group calls (int32 dodge)")
    # ---- expansion alone (fwd copy; backward reduction measured in-graph)
    cell("gqa-expand", lambda: k8.repeat_interleave(rep, dim=1),
         (1, H_Q, s, D))

    # ---- anchor branch: mask built inside a guarded closure (its
    # allocation itself OOMs at 1M — that is a result, not a crash)
    def masked_anchor():
        key_pos = torch.arange(s, device=dev)
        row_mask = (key_pos.unsqueeze(0) <= sel.unsqueeze(1)) \
            .view(1, 1, n_sel, s)
        return F.scaled_dot_product_attention(
            q[:, :, sel], k8.repeat_interleave(rep, dim=1),
            v8.repeat_interleave(rep, dim=1),
            attn_mask=row_mask, scale=scaling)
    cell("anchor-masked", masked_anchor, (1, H_Q, n_sel, D))

    def forced(backend):
        def fn():
            key_pos = torch.arange(s, device=dev)
            row_mask = (key_pos.unsqueeze(0) <= sel.unsqueeze(1)) \
                .view(1, 1, n_sel, s)
            with sdpa_kernel(backends=[backend]):
                return F.scaled_dot_product_attention(
                    q[:, :, sel], k8.repeat_interleave(rep, dim=1),
                    v8.repeat_interleave(rep, dim=1),
                    attn_mask=row_mask, scale=scaling)
        return fn
    cell("anchor-flash!", forced(SDPBackend.FLASH_ATTENTION),
         (1, H_Q, n_sel, D), expect_raise=True,
         note="raises => mask blocks flash")
    cell("anchor-mathonly", forced(SDPBackend.MATH), (1, H_Q, n_sel, D))
    cell("anchor-efficient", forced(SDPBackend.EFFICIENT_ATTENTION),
         (1, H_Q, n_sel, D))

    # ---- duplication-free reformulation: gather IN-GRAPH (symmetric with
    # anchor-masked: both pay gather fwd + scatter bwd into q.grad)
    def flexrow_mask(b, h, q_idx, kv_idx):
        return kv_idx <= sel[q_idx]
    # _compile=True: the eager path materializes the full n_sel x s bool
    # mask plus an O(n_sel*s) block-sum intermediate (~17GB+ at 1M) before
    # compressing — same class of OOM that killed the 07-22 long ladder on
    # the causal mask (fixed in flex_delta.get_block_mask the same way)
    bm_row = create_block_mask(flexrow_mask, B=None, H=None,
                               Q_LEN=n_sel, KV_LEN=s, device=dev,
                               _compile=True)
    cell("anchor-flexrow",
         lambda: _get_flex()(q[:, :, sel], k8, v8, block_mask=bm_row,
                             scale=scaling, enable_gqa=True),
         (1, H_Q, n_sel, D),
         note="gathered in-graph, GQA-native, prefixes via mask")

    # ---- correction alone (production gradient path shape)
    sparse_out = torch.randn(1, s, H_Q, D, device=dev, dtype=dt,
                             requires_grad=True)
    anchor_out = torch.randn(1, n_sel, H_Q, D, device=dev, dtype=dt,
                             requires_grad=True)
    n_anchor = idx.numel()
    idx_dev = idx.to(dev)

    def correction():
        delta = anchor_out[:, :n_anchor] - sparse_out[:, idx_dev]
        corrected = (sparse_out[:, :s_p].reshape(1, n_anchor, GAMMA, H_Q, D)
                     + delta.reshape(1, n_anchor, 1, H_Q, D)) \
            .reshape(1, s_p, H_Q, D)
        return torch.cat((corrected, anchor_out[:, n_anchor:]), dim=1)
    leaves_c = [sparse_out, anchor_out]
    old_leaves = leaves[:]
    leaves.clear()
    leaves.extend(leaves_c)
    cell("correction", correction, (1, s, H_Q, D))
    leaves.clear()
    leaves.extend(old_leaves)

    # ---- MTP head-read cost: ONE query over an s-length cache (the
    # drafting cost per token: dense reads everything; delta reads
    # sink+window, plus one dense read per GAMMA confirmed tokens —
    # amortized delta cost = ((GAMMA-1)*delta + 1*dense)/GAMMA)
    q1 = torch.randn(1, H_Q, 1, D, device=dev, dtype=dt)
    cell("headread-dense",
         lambda: F.scaled_dot_product_attention(
             q1, k8, v8, scale=scaling, enable_gqa=True),
         (1, H_Q, 1, D), backward=False)
    if s > SINK + WINDOW:
        cell("headread-delta",
             lambda: F.scaled_dot_product_attention(
                 q1, torch.cat([k8[:, :, :SINK], k8[:, :, -WINDOW:]], dim=2),
                 torch.cat([v8[:, :, :SINK], v8[:, :, -WINDOW:]], dim=2),
                 scale=scaling, enable_gqa=True),
             (1, H_Q, 1, D), backward=False,
             note=f"amortized = ({GAMMA-1}*this + headread-dense)/{GAMMA}")

    # ---- WHOLE-FUNCTION cells (Jeff 07-21: "init random inputs and call
    # this one function with fwd/bwd"). full-delta-current = production
    # delta_forward_train verbatim; its OOM at long lengths is itself the
    # result (mask + expansion + scores). full-delta-flexrow = the same
    # math with the anchor branch reformulated (gathered rows, GQA-native,
    # prefixes via block mask) and in-graph expansion only for the sparse
    # branch. full-dense = FA2-class causal reference via sdpa.
    from delta_attention.train.flex_delta import delta_forward_train

    cell("full-delta-current",
         lambda: delta_forward_train(q, k8, v8, gamma=GAMMA, window=WINDOW,
                                     sink=SINK),
         (1, s, H_Q, D), note="production composition, verbatim")

    def full_delta_flexrow():
        kr = k8.repeat_interleave(rep, dim=1)
        vr = v8.repeat_interleave(rep, dim=1)
        sparse = _get_flex()(q, kr, vr, block_mask=bm, scale=scaling)
        sparse = sparse.transpose(1, 2)
        dense_sel = _get_flex()(q[:, :, sel], k8, v8, block_mask=bm_row,
                                scale=scaling, enable_gqa=True)             .transpose(1, 2)
        n_a = idx.numel()
        delta = dense_sel[:, :n_a] - sparse[:, idx_dev]
        corrected = (sparse[:, :s_p].reshape(1, n_a, GAMMA, H_Q, D)
                     + delta.reshape(1, n_a, 1, H_Q, D))             .reshape(1, s_p, H_Q, D)
        return torch.cat((corrected, dense_sel[:, n_a:]), dim=1)
    cell("full-delta-flexrow", full_delta_flexrow, (1, s, H_Q, D),
         note="anchor branch reformulated (flexrow), same math")

    if needs_hc:
        # same composition with the sparse branch head-chunked (the anchor
        # branch's n_sel x s shapes stay far below the int32 limit) — the
        # cell that carries the whole-function delta number to 1M
        def full_delta_flexrow_hc():
            sparse = flex_hc(q, k8.repeat_interleave(rep, dim=1),
                             v8.repeat_interleave(rep, dim=1), bm) \
                .transpose(1, 2)
            dense_sel = _get_flex()(q[:, :, sel], k8, v8, block_mask=bm_row,
                                    scale=scaling, enable_gqa=True) \
                .transpose(1, 2)
            n_a = idx.numel()
            delta = dense_sel[:, :n_a] - sparse[:, idx_dev]
            corrected = (sparse[:, :s_p].reshape(1, n_a, GAMMA, H_Q, D)
                         + delta.reshape(1, n_a, 1, H_Q, D)) \
                .reshape(1, s_p, H_Q, D)
            return torch.cat((corrected, dense_sel[:, n_a:]), dim=1)
        cell(f"full-delta-flexrow-hc{HC}", full_delta_flexrow_hc,
             (1, s, H_Q, D),
             note=f"flexrow composition, sparse branch in {HC} head chunks")

    cell("full-dense",
         lambda: F.scaled_dot_product_attention(
             q, k8, v8, is_causal=True, scale=scaling,
             enable_gqa=True).transpose(1, 2),
         (1, s, H_Q, D), note="causal dense reference")

    del q, k8, v8, sparse_out, anchor_out, q1
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-lens", type=str, default="8192,32768")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--out", type=str, default="results/anchorbench.csv")
    args = p.parse_args()

    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"anchorbench_{args.seq_lens.replace(',', '-')}",
                     config=vars(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    written = 0
    # flush after EVERY length — the 07-22 524K crash cost the completed
    # 131K/262K rows (they survived only in the wandb summary)
    with out_path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if fh.tell() == 0:
            w.writerow(["component", "seq_len", "scope", "fwd_ms",
                        "fwdbwd_ms", "note"])
        for s in [int(x) for x in args.seq_lens.split(",")]:
            bench_seq_len(s, args.warmup, args.iters, rows, run)
            w.writerows(rows[written:])
            fh.flush()
            written = len(rows)
            sm, sm_max, temp = gpu_state()
            print(f"[anchorbench] post-{s} clocks {sm}/{sm_max}MHz {temp}C",
                  flush=True)
    run.finish()
    print("[anchorbench] DONE (all numbers PER LAYER; multiply by "
          "n_layers for per-step)", flush=True)


if __name__ == "__main__":
    main()
