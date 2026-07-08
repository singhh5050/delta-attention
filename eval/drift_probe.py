"""Position-resolved inter-anchor delta drift across long documents.

For each (document, context length, layer): compute the delta (dense anchor
row minus sparse row) at every gamma-th anchor and record the cosine between
consecutive anchor deltas BY POSITION — the curve of "how fast the correction
drifts" across document depth. This is the design input for adaptive stride
(does gamma have a positional prior?) and the baseline for WP-2's
smoothness-training claim.

Standalone: no server, no RULER. The carrier forward runs under FA2 (fast,
no logits — LlamaModel only); per-layer q/k/v are captured via hooks and the
delta math uses the T13-validated flex mask (sparse) + the paper's own
query-sparse Triton kernel (dense anchors).

    python eval/drift_probe.py --context-lengths 32768,65536,131072 --n-docs 3

Outputs: results/drift_probe/probe.json (all curves) + wandb line series per
layer + a printed first/middle/last-third summary per context length.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context-lengths", type=str, default="32768,65536,131072")
    p.add_argument("--n-docs", type=int, default=3)
    p.add_argument("--gamma", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--out", type=str, default="results/drift_probe/probe.json")
    return p.parse_args()


def load_model():
    from delta_attention.config import Config
    from delta_attention.sample import init_model

    cfg = Config()
    cfg.attn_implementation = "window"
    cfg.mode = "delta"
    cfg.attn_implementation_original = cfg.attn_implementation
    model, tokenizer = init_model(cfg)
    model.config._attn_implementation = "flash_attention_2"  # carrier only
    model.config.use_cache = False
    model.eval()
    return model.cuda(), tokenizer


def long_docs(tokenizer, min_tokens: int, n_docs: int):
    """PG19 TEST split (no overlap with WP-2 training) docs >= min_tokens."""
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    out = []
    for doc in ds:
        toks = tokenizer.encode(doc["text"], add_special_tokens=False)
        if len(toks) >= min_tokens:
            out.append((doc.get("short_book_title", "?")[:60], toks))
            if len(out) == n_docs:
                return out
    raise SystemExit(f"only {len(out)} test docs with >= {min_tokens} tokens")


@torch.no_grad()
def layer_drift_curve(attn, rotary, normed_hidden, gamma, window, scaling):
    """Cos between consecutive anchor deltas, by anchor position.
    Returns [n_anchors-1] python floats (mean over heads)."""
    from delta_attention.delta_kernel import attention as qsa_kernel
    from delta_attention.llama import apply_rotary_pos_emb, repeat_kv
    from delta_attention.train.flex_delta import _get_flex, anchor_layout, get_block_mask

    hs = normed_hidden.shape[:-1]
    q = attn.q_proj(normed_hidden).view(*hs, -1, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(normed_hidden).view(*hs, -1, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(normed_hidden).view(*hs, -1, attn.head_dim).transpose(1, 2)
    s = q.size(2)
    pos = torch.arange(s, device=q.device).unsqueeze(0)
    cos_e, sin_e = rotary(v.transpose(1, 2), pos)
    q, k = apply_rotary_pos_emb(q, k, cos_e, sin_e)
    k = repeat_kv(k, attn.num_key_value_groups)
    v = repeat_kv(v, attn.num_key_value_groups)

    bm = get_block_mask(s, window, 1024, str(q.device))
    sparse = _get_flex()(q, k, v, block_mask=bm, scale=scaling).transpose(1, 2)

    idx, _, _ = anchor_layout(s, gamma)
    idx = idx.to(q.device)
    dense = qsa_kernel(q[:, :, idx], k, v, idx.unsqueeze(0), scaling).transpose(1, 2)

    deltas = dense - sparse[:, idx]  # [1, n, h, d]
    c = torch.nn.functional.cosine_similarity(
        deltas[:, :-1].float(), deltas[:, 1:].float(), dim=-1)  # [1, n-1, h]
    return [round(x, 5) for x in c.mean(dim=-1)[0].tolist()]


def main():
    args = parse_args()
    lengths = [int(x) for x in args.context_lengths.split(",")]
    for s in lengths:
        assert s % 64 == 0, s

    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"drift_probe_g{args.gamma}", config=vars(args))

    model, tokenizer = load_model()
    base = model.model  # LlamaModel: no lm_head, no logits
    layers = base.layers
    rotary = base.rotary_emb
    scaling = layers[0].self_attn.head_dim ** -0.5
    docs = long_docs(tokenizer, max(lengths), args.n_docs)
    print(f"[probe] docs: {[t for t, _ in docs]}", flush=True)

    results = {}  # f"{doc_i}/{s}/{layer}" -> [cos...]
    curves = {}

    def make_hook(layer_i, store):
        def fn(mod, inp, out):
            store[layer_i] = layer_drift_curve(
                layers[layer_i].self_attn, rotary, out, args.gamma, args.window, scaling)
        return fn

    for di, (title, toks) in enumerate(docs):
        for s in lengths:
            ids = torch.tensor([toks[:s]], dtype=torch.long, device="cuda")
            store = {}
            handles = [l.input_layernorm.register_forward_hook(make_hook(i, store))
                       for i, l in enumerate(layers)]
            with torch.no_grad():
                base(ids)
            for h in handles:
                h.remove()
            for li, curve in store.items():
                results[f"{di}/{s}/{li}"] = curve
                curves.setdefault((s, li), []).append(curve)
            def thirds(v):
                t = max(len(v) // 3, 1)
                return (sum(v[:t]) / t, sum(v[t:2 * t]) / t,
                        sum(v[2 * t:]) / max(len(v) - 2 * t, 1))

            # per-position mean over layers -> first/mid/last-third summary
            n = min(len(c) for c in store.values())
            posmean = [sum(store[i][p] for i in store) / len(store) for p in range(n)]
            f, m, l = thirds(posmean)
            allc = [x for c in store.values() for x in c]
            print(f"[probe] doc{di} '{title}' s={s}: mean cos by thirds "
                  f"first={f:.3f} mid={m:.3f} last={l:.3f} (overall {sum(allc)/len(allc):.3f})",
                  flush=True)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": vars(args), "curves": results}))
    print(f"[probe] wrote {out_path}", flush=True)

    # wandb: per (length, layer) mean curve across docs, as line series
    for (s, li), doc_curves in sorted(curves.items()):
        n = min(len(c) for c in doc_curves)
        mean_curve = [sum(c[p] for c in doc_curves) / len(doc_curves) for p in range(n)]
        step_axis = [p * args.gamma for p in range(n)]
        table = wandb.Table(data=[[x, y] for x, y in zip(step_axis, mean_curve)],
                            columns=["token_position", "interanchor_cos"])
        run.log({f"drift_curve/s{s}/layer_{li:02d}":
                 wandb.plot.line(table, "token_position", "interanchor_cos",
                                 title=f"s={s} layer {li}")})
    artifact = wandb.Artifact("drift_probe_curves", type="probe")
    artifact.add_file(str(out_path))
    run.log_artifact(artifact)
    run.finish()
    print("[probe] DONE", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    main()
