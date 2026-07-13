"""LongBench v1 QA (generation+F1) and LongBench v2 MCQ (logprob) under the pipeline.

Jeff's post-training eval ask: run the trained adapters on benchmarks beyond
RULER. Four arms share identical prompts/samples; only the attention path and
checkpoint differ:

    base_dense   flash_attention_2, mode=none (dense ceiling)
    base_delta   window/delta gamma=64, no adapter
    ce_delta     window/delta gamma=64, adapter checkpoints/pilot_delta
    dense_delta  window/delta gamma=64, adapter checkpoints/pilot_dense

v1 tasks are the long-context short-generation QA subset (avg contexts well
past sink+window=3072, so the delta path is active). v2 is scored by argmax
over the logprobs of the four choice letters — no generation, no string-match
noise (the MMLU-style scoring, but at lengths where delta != dense).

    python eval/longbench_eval.py --suite v1 --n-samples 50
    python eval/longbench_eval.py --suite v2 --n-samples 200
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import string
import sys
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

try:  # scorer/truncation stay importable on torch-less boxes (offline tests)
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MAX_PROMPT_TOKENS = 31_500

# LongBench official templates/limits (config/dataset2prompt.json,
# dataset2maxlen.json in THUDM/LongBench).
QA_TEMPLATE = (
    "Answer the question based on the given passages. Only give me the answer "
    "and do not output any other words.\n\nThe following are given passages.\n"
    "{context}\n\nAnswer the question based on the given passages. Only give "
    "me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
)
MULTIFIELD_TEMPLATE = (
    "Read the following text and answer briefly.\n\n{context}\n\nNow, answer "
    "the following question based on the above text, only give me the answer "
    "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
)
V1_TASKS = {
    "hotpotqa": (QA_TEMPLATE, 32),
    "2wikimqa": (QA_TEMPLATE, 32),
    "musique": (QA_TEMPLATE, 32),
    "multifieldqa_en": (MULTIFIELD_TEMPLATE, 64),
}

V2_TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\n"
    "Answer with a single letter (A, B, C, or D)."
)

ARMS = ("base_dense", "base_delta", "ce_delta", "dense_delta",
        # decode arms (base model, delta prefill g64, non-dense decode):
        # does the RULER decode cliff replicate on QA, or is it
        # needle-retrieval-specific?
        "sparse_dec", "delta_dec2", "delta_dec16")
ADAPTERS = {"ce_delta": "checkpoints/pilot_delta",
            "dense_delta": "checkpoints/pilot_dense"}
DECODE_ARMS = {"sparse_dec": ("sparse", None),
               "delta_dec2": ("delta", 2),
               "delta_dec16": ("delta", 16)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["v1", "v2"], required=True)
    p.add_argument("--arms", type=str, default=",".join(ARMS))
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--gamma", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--out", type=str, default="results/longbench.csv")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LongBench qa_f1_score (metrics.py), inlined
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gt = normalize_answer(ground_truth).split()
    common = Counter(pred) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truths) -> float:
    return max(f1_score(prediction, gt) for gt in ground_truths)


# ---------------------------------------------------------------------------
# data (hub files directly -- THUDM/LongBench is a script dataset, which
# newer `datasets` refuses to load; the raw jsonl needs no script)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)  # called once per arm — don't re-parse the zip 4x
def load_v1(task: str):
    from huggingface_hub import hf_hub_download

    zpath = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(zpath) as z, z.open(f"data/{task}.jsonl") as f:
        return [json.loads(line) for line in f]


def load_v2():
    from huggingface_hub import hf_hub_download

    jpath = hf_hub_download("THUDM/LongBench-v2", "data.json", repo_type="dataset")
    return json.loads(Path(jpath).read_text())


def truncate_middle(prompt: str, tokenizer, max_tokens: int) -> str:
    """LongBench's scheme: keep the head and tail halves of the token stream."""
    toks = tokenizer.encode(prompt, add_special_tokens=False)
    if len(toks) <= max_tokens:
        return prompt
    half = max_tokens // 2
    return (tokenizer.decode(toks[:half], skip_special_tokens=True)
            + tokenizer.decode(toks[-half:], skip_special_tokens=True))


def chat_wrap(prompt: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def build_model(arm: str, gamma: int, window: int):
    from delta_attention.config import Config
    from delta_attention.sample import init_model

    cfg = Config()
    if arm == "base_dense":
        cfg.attn_implementation = "flash_attention_2"
        cfg.mode = "none"
    else:
        cfg.attn_implementation = "window"
        cfg.mode = "delta"
        cfg.delta_lambda = gamma
        cfg.sliding_window = window
        if arm in DECODE_ARMS:
            cfg.decode_mode, gd = DECODE_ARMS[arm]
            if gd is not None:
                cfg.gamma_dec = gd
                cfg.refresh_policy = "fixed"
    cfg.attn_implementation_original = cfg.attn_implementation
    if arm in ADAPTERS:
        path = Path(ADAPTERS[arm])
        assert path.exists(), f"adapter missing: {path} (download from wandb first)"
        cfg.checkpoint = str(path)
    model, tokenizer = init_model(cfg)
    model.config.log_drift = False
    model.eval().cuda()
    return model, tokenizer


# NOTE: freeing between arms happens inline in main() — `del model` only
# works from the scope that owns the binding; a free_model(model) helper's
# local del is a no-op and leaves two 16GB models resident during the next
# build_model (OOM on 40GB boxes).


def generate_answer(model, tokenizer, prompt: str, max_new: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = tokenizer.decode(ids[0, inputs["input_ids"].size(1):],
                            skip_special_tokens=True)
    return text.strip().split("\n")[0]


def choice_logprobs(model, tokenizer, prompt: str, letter_ids) -> int:
    """Single prefill forward; argmax over the four letter tokens at the last
    position. Mirrors _sample's prefill exactly (no_lm_head + manual lm_head)."""
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False)["input_ids"].cuda()
    setattr(model, "no_lm_head", True)
    if model.config.mode in ["delta", "recompute", "sparse-only"]:
        model.config._attn_implementation = model.config.attn_implementation_original
    with torch.no_grad():
        out = model(ids, use_cache=True)  # mirrors _sample's prefill call
        hidden = out.logits[:, -1, :]  # hidden states under no_lm_head
        logits = model.lm_head(hidden).float()[0]
        pick = int(torch.argmax(logits[letter_ids]).item())
    del out
    return pick


# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------

def run_v1(model, tokenizer, arm: str, n: int, log):
    scores = {}
    for task, (template, max_new) in V1_TASKS.items():
        rows = load_v1(task)[:n]
        if not rows:
            raise SystemExit(f"no samples for {task} (n={n})")
        vals = []
        for i, ex in enumerate(rows):
            prompt = template.format(context=ex["context"], input=ex["input"])
            prompt = truncate_middle(prompt, tokenizer, MAX_PROMPT_TOKENS)
            pred = generate_answer(model, tokenizer,
                                   chat_wrap(prompt, tokenizer), max_new)
            vals.append(qa_f1_score(pred, ex["answers"]))
            if i < 2:
                print(f"[lb-v1] {arm}/{task} #{i}: pred={pred!r} "
                      f"gt={ex['answers']} f1={vals[-1]:.2f}", flush=True)
        scores[task] = sum(vals) / len(vals)
        log(arm, "v1", task, len(vals), scores[task])
        print(f"[lb-v1] {arm}/{task}: F1 {scores[task]:.4f} (n={len(vals)})",
              flush=True)
    return scores


def select_v2(tokenizer, n: int):
    """Deterministic: 'short' samples in dataset order whose full prompt fits."""
    picked = []
    for ex in load_v2():
        if ex.get("length") != "short":
            continue
        prompt = V2_TEMPLATE.format(context=ex["context"], question=ex["question"],
                                    A=ex["choice_A"], B=ex["choice_B"],
                                    C=ex["choice_C"], D=ex["choice_D"])
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > MAX_PROMPT_TOKENS:
            continue
        picked.append((prompt, ex["answer"], ex["difficulty"]))
        if len(picked) == n:
            break
    return picked


def run_v2(model, tokenizer, arm: str, samples, log):
    letters = ["A", "B", "C", "D"]
    letter_ids = torch.tensor(
        [tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]).cuda()
    hits, by_diff = [], {}
    for prompt, answer, diff in samples:
        pick = choice_logprobs(model, tokenizer, chat_wrap(prompt, tokenizer),
                               letter_ids)
        ok = letters[pick] == answer
        hits.append(ok)
        by_diff.setdefault(diff, []).append(ok)
    acc = sum(hits) / len(hits)
    log(arm, "v2", "mcq", len(hits), acc)
    for diff, xs in sorted(by_diff.items()):
        log(arm, "v2", f"mcq_{diff}", len(xs), sum(xs) / len(xs))
    print(f"[lb-v2] {arm}: acc {acc:.4f} (n={len(hits)}) "
          f"{ {d: round(sum(x)/len(x), 3) for d, x in by_diff.items()} }", flush=True)
    return acc


def main():
    args = parse_args()
    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                     name=f"longbench_{args.suite}", config=vars(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    fh = out_path.open("a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["suite", "task", "arm", "n", "score", "run_id"])

    def log(arm, suite, task, n, score):
        # run_id disambiguates rows when a retried chain appends a second set
        writer.writerow([suite, task, arm, n, f"{score:.4f}", run.id])
        fh.flush()
        run.log({f"longbench/{arm}/{suite}_{task}": score})
        run.summary[f"{suite}_{task}_{arm}"] = score

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:  # fail before any model is built, not after arm 1's GPU-hours
        raise SystemExit(f"unknown arms {unknown}; valid: {ARMS}")

    v2_samples = None
    for arm in arms:
        model, tokenizer = build_model(arm, args.gamma, args.window)
        if args.suite == "v1":
            run_v1(model, tokenizer, arm, args.n_samples, log)
        else:
            if v2_samples is None:
                v2_samples = select_v2(tokenizer, args.n_samples)
                print(f"[lb-v2] selected {len(v2_samples)} samples", flush=True)
                if not v2_samples:
                    raise SystemExit("no LongBench-v2 samples fit the token budget")
            run_v2(model, tokenizer, arm, v2_samples, log)
        # break the _sample MethodType reference cycle, then drop OUR binding —
        # del must run in this scope or the object survives into the next load
        model._sample = None
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    fh.close()
    run.finish()
    print("[longbench] DONE", flush=True)


if __name__ == "__main__":
    main()
