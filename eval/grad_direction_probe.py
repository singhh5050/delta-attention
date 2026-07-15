"""Directional-bias probe (Jeff's q/k/v concern, beyond magnitude).

"Even if the norm is stable, it could be biasing the q,k,v projections to
learn much more information about the gamma rows than the other rows."

For a trained adapter, at the trained weights, compute per projection W:
  g_anchor = dL/dW from ANCHOR-token losses only (labels masked elsewhere)
  g_window = dL/dW from non-anchor (sliding-window-row) losses only
  dW       = the LoRA update B@A * (alpha/r), read from the adapter file
and report norms, cos(g_anchor, g_window), cos(g_*, dW), and the projection
of dW onto each gradient direction — per q/k/v/o at layers 0/8/16/24/31.

If training were directionally dominated by anchor rows, dW should align
with g_anchor much more than with g_window.

    python eval/grad_direction_probe.py --arms delta,delta_32k,detach,dense
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LAYERS = (0, 8, 16, 24, 31)
PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")


def lora_updates(adapter_dir):
    """dW = B @ A * (alpha/r) per (layer, proj), from the adapter file."""
    import json

    from safetensors.torch import load_file

    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    out = {}
    for li in LAYERS:
        for pj in PROJS:
            stem = f"model.layers.{li}.self_attn.{pj}"
            a = next((v for k, v in sd.items() if stem in k and "lora_A" in k), None)
            b = next((v for k, v in sd.items() if stem in k and "lora_B" in k), None)
            if a is not None and b is not None:
                out[(li, pj)] = (b.float() @ a.float()) * scale
    return out


def masked_labels(ids, gamma, anchor: bool):
    """Anchor rows follow delta_forward_train's layout: every gamma-th row of
    the uniform region + the dense tail. anchor=True keeps only those;
    False keeps only the window rows. -100 elsewhere.

    SHIFT-AWARE: chunked_ce_hidden computes the loss of hidden position p
    from labels[p+1], so to capture the loss AT anchor rows idx we must keep
    labels at idx+1 (masking at idx would credit the row BEFORE each anchor
    — a window row — and swap the two gradient populations)."""
    from delta_attention.train.flex_delta import anchor_layout

    s = ids.size(1)
    idx, tail, _ = anchor_layout(s, gamma)
    pos = torch.cat((idx, tail)) + 1  # label p+1 carries hidden row p's loss
    pos = pos[pos < s]
    mask = torch.zeros(s, dtype=torch.bool)
    mask[pos] = True
    labels = ids.clone()
    labels[0, ~mask if anchor else mask] = -100
    return labels


def probe_arm(arm, adapter_dir, batches, gamma, window, log):
    from delta_attention.config import Config
    from delta_attention.sample import init_model
    from delta_attention.train.train_delta import chunked_ce_hidden

    cfg = Config()
    cfg.attn_implementation = "window"
    cfg.mode = "delta"
    cfg.delta_lambda = gamma
    cfg.sliding_window = window
    cfg.attn_implementation_original = cfg.attn_implementation
    if adapter_dir:
        cfg.checkpoint = adapter_dir  # merged -> gradients land on W itself
    model, tokenizer = init_model(cfg)
    model.config.log_drift = False
    model.config.detach_delta = False
    model.config._attn_implementation = "flex_delta_train"
    model.config.use_cache = False
    model.eval().cuda()
    setattr(model, "no_lm_head", True)  # loss via chunked_ce_hidden

    # freeze EVERYTHING first — default requires_grad=True on all 8B params
    # would allocate ~64GB of fp32 grads on backward
    for prm in model.parameters():
        prm.requires_grad_(False)
    targets = {}
    for li in LAYERS:
        attn = model.model.layers[li].self_attn
        for pj in PROJS:
            w = getattr(attn, pj).weight
            w.requires_grad_(True)
            targets[(li, pj)] = w

    dW = lora_updates(adapter_dir) if adapter_dir else {}

    grads = {"anchor": {}, "window": {}}
    for which in ("anchor", "window"):
        for w in targets.values():
            w.grad = None
        for ids in batches:
            labels = masked_labels(ids, gamma, anchor=(which == "anchor")).cuda()
            out = model(input_ids=ids.cuda())
            loss = chunked_ce_hidden(out.logits, model.lm_head.weight, labels)
            loss.backward()
            del out
        for key, w in targets.items():
            grads[which][key] = w.grad.detach().float().clone()
        torch.cuda.empty_cache()

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0).item()

    for key in targets:
        li, pj = key
        ga, gw = grads["anchor"][key], grads["window"][key]
        row = {"arm": arm, "layer": li, "proj": pj,
               "norm_anchor": ga.norm().item(), "norm_window": gw.norm().item(),
               "cos_anchor_window": cos(ga, gw)}
        if key in dW:
            d = dW[key].cuda()
            row["cos_dW_anchor"] = cos(d, ga)
            row["cos_dW_window"] = cos(d, gw)
            # projection coefficients of dW onto each (unit) gradient
            row["proj_dW_anchor"] = (d.flatten() @ ga.flatten()
                                     / ga.norm().clamp_min(1e-12)).item()
            row["proj_dW_window"] = (d.flatten() @ gw.flatten()
                                     / gw.norm().clamp_min(1e-12)).item()
        log(row)
        print(f"[gradprobe] {arm} L{li} {pj}: |gA|/|gW| "
              f"{row['norm_anchor'] / max(row['norm_window'], 1e-12):.3f} "
              f"cos(gA,gW) {row['cos_anchor_window']:.3f}"
              + (f" cos(dW,gA) {row['cos_dW_anchor']:.3f}"
                 f" cos(dW,gW) {row['cos_dW_window']:.3f}" if key in dW else ""),
              flush=True)

    model._sample = None
    del model, targets, grads
    gc.collect()
    torch.cuda.empty_cache()
    return tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", type=str, default="delta,delta_32k,detach,dense")
    p.add_argument("--n-batches", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--gamma", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--adapters-root", type=str, default="checkpoints")
    p.add_argument("--out", type=str, default="results/grad_direction.csv")
    args = p.parse_args()

    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name="grad_direction_probe", config=vars(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["arm", "layer", "proj", "norm_anchor", "norm_window",
            "cos_anchor_window", "cos_dW_anchor", "cos_dW_window",
            "proj_dW_anchor", "proj_dW_window"]
    new = not out_path.exists()
    fh = out_path.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols + ["run_id"])
    if new:
        w.writeheader()

    def log(row):
        w.writerow({**{c: row.get(c, "") for c in cols}, "run_id": run.id})
        fh.flush()
        run.log({f"gradprobe/{row['arm']}/L{row['layer']}_{row['proj']}_{k}": v
                 for k, v in row.items() if isinstance(v, float)})

    batches = None
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        adapter = "" if arm == "base" else str(Path(args.adapters_root) / f"pilot_{arm}")
        if adapter and not Path(adapter).exists():
            raise SystemExit(f"adapter missing: {adapter}")
        if batches is None:
            import transformers
            tok = transformers.AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct")
            # held-out: PG19 TEST split (training used train split)
            from eval.ppl_eval import test_chunks
            batches = test_chunks(tok, args.seq_len, args.n_batches)
        probe_arm(arm, adapter, batches, args.gamma, args.window, log)

    fh.close()
    run.finish()
    print("[gradprobe] DONE", flush=True)


if __name__ == "__main__":
    main()
