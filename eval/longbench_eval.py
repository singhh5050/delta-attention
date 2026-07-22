"""LongBench v1 QA (generation+F1), LongBench v2 MCQ and InfiniteBench En.MC
(letter-logprob) under the pipeline.

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
    python eval/longbench_eval.py --suite enmc --n-samples 229
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

# Full English LongBench v1 (Jeff, 07-21: "maybe with all of the LongBench
# tasks"). NOTHING here is transcribed: templates, generation lengths, and
# metric functions load at runtime from the official THUDM/LongBench files
# vendored verbatim in third_party/LongBench (commit pinned in
# LongBench.lock) — the same never-reimplement pattern as ruler_client.py.
# The only protocol facts encoded locally are the two task-name sets below,
# which live in official control flow that cannot be imported:
V1_FULL_EN = [  # the 16 English tasks (LongBench README, "en" column)
    "narrativeqa", "qasper", "multifieldqa_en",          # single-doc QA
    "hotpotqa", "2wikimqa", "musique",                   # multi-doc QA
    "gov_report", "qmsum", "multi_news",                 # summarization
    "trec", "triviaqa", "samsum",                        # few-shot
    "passage_count", "passage_retrieval_en",             # synthetic
    "lcc", "repobench-p",                                # code
]
# pred.py: build_chat skipped for these ("chat models are better off without
# build prompts on these tasks") — few-shot and code completion run raw
NO_CHAT_WRAP = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
# eval.py scorer: prediction.lstrip('\n').split('\n')[0] applied ONLY here;
# all other tasks score the raw generation (run_v1 above predates this and
# takes the first line on QA tasks — v1full follows the official scorer)
FIRST_LINE_ONLY = {"trec", "triviaqa", "samsum", "lsht"}

LB_OFFICIAL_DIR = REPO_ROOT / "third_party" / "LongBench"


@lru_cache(maxsize=None)
def lb_official():
    """(dataset2prompt, dataset2maxlen, dataset2metric) from the vendored
    official files. eval.py is loaded under a private module name (our repo
    has an eval/ package, and `eval` shadows it); its `from metrics import
    ...` resolves against the vendored dir, which needs jieba/fuzzywuzzy/
    rouge installed (chain setup does this for the lbfull stage)."""
    import importlib.util

    prompts = json.loads((LB_OFFICIAL_DIR / "dataset2prompt.json").read_text())
    maxlens = json.loads((LB_OFFICIAL_DIR / "dataset2maxlen.json").read_text())
    sys.path.insert(0, str(LB_OFFICIAL_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "_lb_official_eval", LB_OFFICIAL_DIR / "eval.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(LB_OFFICIAL_DIR))
    missing = [t for t in V1_FULL_EN
               if t not in prompts or t not in maxlens or t not in mod.dataset2metric]
    assert not missing, f"vendored LongBench files lack tasks: {missing}"
    return prompts, maxlens, mod.dataset2metric

# LongBench gov_report (dataset2prompt/dataset2maxlen: summarization, 512 new
# tokens). Jeff's speculative-decoding probe: long natural-language generation
# where locally-predictable tokens dominate — the regime where a sparse/delta
# draft should hold up, unlike RULER's impossible-by-construction retrieval.
GOVREPORT_TEMPLATE = (
    "You are given a report by a government agency. Write a one-page summary "
    "of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of "
    "the report.\n\nSummary:"
)
GOVREPORT_MAX_NEW = 512

V2_TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\n"
    "Answer with a single letter (A, B, C, or D)."
)

# MMLU, 0-shot letter-logprob. Prompts sit far inside sink+window (3072), so
# delta == dense mathematically — this does NOT measure delta adaptation; it
# measures whether finetuning damaged general capabilities (Jeff's
# "we ruined the posttraining" concern), on a protocol consistent across arms.
MMLU_TEMPLATE = (
    "The following is a multiple choice question. Answer with a single letter "
    "(A, B, C, or D).\n\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nAnswer:"
)

# InfiniteBench longbook_choice_eng prompt (prompt.py in the official repo)
ENMC_TEMPLATE = (
    "Read the book and answer the question.\n\n{context}\n\n"
    "Question: {question}\n\nOnly one of the following options is correct, "
    "tell me the answer using one single letter (A, B, C, or D). Don't say "
    "anything else.\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nAnswer:"
)

# DEFAULT_ARMS stays the original seven so existing chain stages (which rely
# on the default --arms) keep working; new adapter arms are opt-in via --arms.
DEFAULT_ARMS = ("base_dense", "base_delta", "ce_delta", "dense_delta",
                # decode arms (base model, delta prefill g64, non-dense decode):
                # does the RULER decode cliff replicate on QA, or is it
                # needle-retrieval-specific?
                "sparse_dec", "delta_dec2", "delta_dec16")
ARMS = DEFAULT_ARMS + ("delta_dec4", "delta_dec8",
                       "distill_delta", "distill_mix_delta", "distill_dft_delta",
                       "ce32k_delta", "dense32k_delta", "detach32k_delta",
                       "distill_dftmix_delta")
ADAPTERS = {"ce_delta": "checkpoints/pilot_delta",
            "dense_delta": "checkpoints/pilot_dense",
            "distill_delta": "checkpoints/pilot_distill",
            "distill_mix_delta": "checkpoints/pilot_distill_mix",
            "distill_dft_delta": "checkpoints/pilot_distill_dft",
            "ce32k_delta": "checkpoints/pilot_delta_32k",
            "dense32k_delta": "checkpoints/pilot_dense_32k",
            "detach32k_delta": "checkpoints/pilot_detach_32k",
            "distill_dftmix_delta": "checkpoints/pilot_distill_dftmix"}
DECODE_ARMS = {"sparse_dec": ("sparse", None),
               "delta_dec2": ("delta", 2),
               "delta_dec4": ("delta", 4),
               "delta_dec8": ("delta", 8),
               "delta_dec16": ("delta", 16)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["v1", "v1full", "v2", "enmc",
                                       "govreport", "mmlu"],
                   required=True)
    p.add_argument("--arms", type=str, default=",".join(DEFAULT_ARMS))
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--gamma", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--out", type=str, default="results/longbench.csv")
    p.add_argument("--force-dense", action="store_true",
                   help="evaluate every arm under plain dense attention "
                        "(T2 capability-retention protocol: does the TRAINED "
                        "model match dense-trained on tasks, independent of "
                        "the sparse pipeline?). Arm labels get an '@dense' "
                        "suffix in the CSVs so rows never collide with "
                        "pipeline-eval rows of the same adapter")
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


def rouge_l_score(prediction: str, ground_truths) -> float:
    """LongBench's rouge_score: rouge-l f via the `rouge` package (installed
    by the specdec chain mode; import stays local so torch-less offline test
    boxes never need it)."""
    from rouge import Rouge

    best = 0.0
    for gt in ground_truths:
        try:
            s = Rouge().get_scores([prediction], [gt], avg=True)["rouge-l"]["f"]
        except Exception:  # LongBench metrics.py uses a bare except here too:
            s = 0.0        # rouge can also raise RecursionError on degenerate
        best = max(best, s)  # period-free generations, not just ValueError
    return best


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


def load_enmc():
    from huggingface_hub import hf_hub_download

    jpath = hf_hub_download("xinrongzhang2022/InfiniteBench",
                            "longbook_choice_eng.jsonl", repo_type="dataset")
    with open(jpath) as f:
        return [json.loads(line) for line in f]


def enmc_correct_letter(ex):
    """InfiniteBench stores the answer as its option text; map it to A-D.
    Returns None when it matches no option (skip the sample, don't guess)."""
    if not ex.get("answer"):
        return None
    ans = ex["answer"][0].strip()
    for letter, opt in zip("ABCD", ex["options"]):
        if opt.strip() == ans:
            return letter
    return None


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

def build_model(arm: str, gamma: int, window: int, force_dense: bool = False):
    from delta_attention.config import Config
    from delta_attention.sample import init_model

    cfg = Config()
    if arm == "base_dense" or force_dense:
        # force_dense: MMLU prompts are shorter than delta_forward's cut_n
        # (its anchor arange crashes on s < ~170), and at these lengths
        # delta == dense anyway — retention is measured under the model's
        # native attention with the arm's adapter
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


def generate_answer(model, tokenizer, prompt: str, max_new: int,
                    first_line_only: bool = True, **gen_kwargs) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             **gen_kwargs)
    text = tokenizer.decode(ids[0, inputs["input_ids"].size(1):],
                            skip_special_tokens=True).strip()
    # short-answer QA: first line only; summarization: keep the whole thing
    return text.split("\n")[0] if first_line_only else text


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

def run_v1(model, tokenizer, arm: str, n: int, log, slog=None):
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
            if slog:
                slog("v1", task, arm, i, pred[:80],
                     " | ".join(ex["answers"])[:80], round(vals[-1], 4))
            if i < 2:
                print(f"[lb-v1] {arm}/{task} #{i}: pred={pred!r} "
                      f"gt={ex['answers']} f1={vals[-1]:.2f}", flush=True)
        scores[task] = sum(vals) / len(vals)
        log(arm, "v1", task, len(vals), scores[task])
        print(f"[lb-v1] {arm}/{task}: F1 {scores[task]:.4f} (n={len(vals)})",
              flush=True)
    return scores


def run_v1full(model, tokenizer, arm: str, n: int, log, slog=None):
    """All 16 English LongBench tasks under the official protocol: vendored
    templates/maxlens/metrics, official chat-wrap and first-line rules, and
    pred.py's samsum eos special-case (newline as extra eos + min_length,
    'prevent illegal output on samsum'). Scoring mirrors eval.py's scorer:
    max over ground truths, all_classes passed through."""
    prompts, maxlens, metrics = lb_official()
    scores = {}
    for task, rows in ((t, load_v1(t)[:n]) for t in V1_FULL_EN):
        if not rows:
            raise SystemExit(f"no samples for {task} (n={n})")
        template, max_new, metric_fn = prompts[task], maxlens[task], metrics[task]
        vals = []
        for i, ex in enumerate(rows):
            prompt = template.format(context=ex["context"], input=ex["input"])
            prompt = truncate_middle(prompt, tokenizer, MAX_PROMPT_TOKENS)
            if task not in NO_CHAT_WRAP:
                prompt = chat_wrap(prompt, tokenizer)
            extra = {}
            if task == "samsum":
                inp_len = len(tokenizer.encode(prompt, add_special_tokens=False))
                extra = {"min_length": inp_len + 1,
                         "eos_token_id": [tokenizer.eos_token_id,
                                          tokenizer.encode("\n", add_special_tokens=False)[-1]]}
            pred = generate_answer(model, tokenizer, prompt, max_new,
                                   first_line_only=False, **extra)
            scored = (pred.lstrip("\n").split("\n")[0]
                      if task in FIRST_LINE_ONLY else pred)
            vals.append(max(metric_fn(scored, gt,
                                      all_classes=ex.get("all_classes"))
                            for gt in ex["answers"]))
            if slog:
                slog("v1full", task, arm, i, pred[:80],
                     " | ".join(ex["answers"])[:80], round(vals[-1], 4))
            if i < 2:
                print(f"[lb-full] {arm}/{task} #{i}: score {vals[-1]:.3f} "
                      f"pred[:80]={pred[:80]!r}", flush=True)
        scores[task] = sum(vals) / len(vals)
        log(arm, "v1full", task, len(vals), scores[task])
        print(f"[lb-full] {arm}/{task}: {scores[task]:.4f} (n={len(vals)})",
              flush=True)
    log(arm, "v1full", "MEAN", len(scores),
        sum(scores.values()) / len(scores))
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


def run_govreport(model, tokenizer, arm: str, n: int, log, slog=None):
    rows = load_v1("gov_report")[:n]
    if not rows:
        raise SystemExit(f"no gov_report samples (n={n})")
    vals = []
    for i, ex in enumerate(rows):
        # .replace, not .format: government reports can contain literal braces
        prompt = GOVREPORT_TEMPLATE.replace("{context}", ex["context"])
        prompt = truncate_middle(prompt, tokenizer, MAX_PROMPT_TOKENS)
        pred = generate_answer(model, tokenizer, chat_wrap(prompt, tokenizer),
                               GOVREPORT_MAX_NEW, first_line_only=False)
        vals.append(rouge_l_score(pred, ex["answers"]))
        if slog:  # per-sample scores or the 7-arm "is there any difference"
            slog("govreport", "gov_report", arm, i, "", "",  # question can't
                 round(vals[-1], 4))                         # be paired-tested
        if i < 2:
            print(f"[govreport] {arm} #{i}: rouge-l {vals[-1]:.3f} "
                  f"pred[:120]={pred[:120]!r}", flush=True)
    score = sum(vals) / len(vals)
    log(arm, "govreport", "gov_report", len(vals), score)
    print(f"[govreport] {arm}: ROUGE-L {score:.4f} (n={len(vals)})", flush=True)
    return score


def select_mmlu(n: int):
    """Deterministic even stride over the full MMLU test split (14,042 rows,
    ordered by subject — a head-slice would be all abstract_algebra; striding
    covers every subject proportionally)."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    step = max(len(ds) // n, 1)
    picked = []
    for i in range(0, len(ds), step):
        ex = ds[i]
        c = ex["choices"]
        # .replace, not .format: math questions can contain literal braces
        prompt = MMLU_TEMPLATE
        for field, val in (("{question}", ex["question"]), ("{A}", c[0]),
                           ("{B}", c[1]), ("{C}", c[2]), ("{D}", c[3])):
            prompt = prompt.replace(field, str(val))
        picked.append((prompt, "ABCD"[ex["answer"]], ex["subject"]))
        if len(picked) == n:
            break
    return picked


def select_enmc(tokenizer, n: int):
    """First n En.MC samples in dataset order (deterministic). Book contexts
    are ~100K+ tokens, so every prompt gets middle-truncated to the 32K eval
    budget — this compares the arms on identical truncated inputs, not
    full-book QA. Truncation happens here, once, not per arm."""
    picked, skipped = [], 0
    for ex in load_enmc():
        letter = enmc_correct_letter(ex)
        if letter is None:
            skipped += 1
            continue
        o = ex["options"]
        prompt = ENMC_TEMPLATE.format(context=ex["context"], question=ex["input"],
                                      A=o[0], B=o[1], C=o[2], D=o[3])
        picked.append((truncate_middle(prompt, tokenizer, MAX_PROMPT_TOKENS),
                       letter, None))
        if len(picked) == n:
            break
    if skipped:
        print(f"[enmc] skipped {skipped} samples whose answer matched no option",
              flush=True)
    return picked


def run_mcq(model, tokenizer, arm: str, suite: str, samples, log, slog=None):
    """Shared letter-logprob runner for the MCQ suites (v2, enmc): one scoring
    path so a fix (e.g. letter tokenization) can't silently diverge between
    suites. samples: (prompt, answer_letter, difficulty_or_None). slog logs
    per-sample correctness — without it only aggregates survive and no paired
    (McNemar-style) test between arms is possible post-hoc."""
    letters = ["A", "B", "C", "D"]
    letter_ids = torch.tensor(
        [tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]).cuda()
    hits, by_diff = [], {}
    for i, (prompt, answer, diff) in enumerate(samples):
        pick = choice_logprobs(model, tokenizer, chat_wrap(prompt, tokenizer),
                               letter_ids)
        ok = letters[pick] == answer
        hits.append(ok)
        if slog:
            slog(suite, "mcq", arm, i, letters[pick], answer, int(ok))
        if diff is not None:
            by_diff.setdefault(diff, []).append(ok)
    acc = sum(hits) / len(hits)
    log(arm, suite, "mcq", len(hits), acc)
    for diff, xs in sorted(by_diff.items()):
        log(arm, suite, f"mcq_{diff}", len(xs), sum(xs) / len(xs))
    print(f"[{suite}] {arm}: acc {acc:.4f} (n={len(hits)})"
          + (f" { {d: round(sum(x)/len(x), 3) for d, x in by_diff.items()} }"
             if by_diff else ""), flush=True)
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

    # per-sample scores for EVERY suite (same dir as --out -> box archive
    # picks it up); enables paired between-arm tests, which aggregates cannot
    # — samples are identical across arms, so pairing is by idx
    samples_path = out_path.with_name(out_path.stem + "_samples.csv")
    new_samples_file = not samples_path.exists()
    sfh = samples_path.open("a", newline="")
    swriter = csv.writer(sfh)
    if new_samples_file:
        swriter.writerow(["suite", "task", "arm", "idx", "pred", "gold",
                          "score", "run_id"])

    def slog(suite, task, arm, idx, pred, gold, score):
        swriter.writerow([suite, task, arm, idx, pred, gold, score, run.id])
        sfh.flush()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:  # fail before any model is built, not after arm 1's GPU-hours
        raise SystemExit(f"unknown arms {unknown}; valid: {ARMS}")

    v2_samples = None
    enmc_samples = None
    mmlu_samples = None
    for arm in arms:
        model, tokenizer = build_model(arm, args.gamma, args.window,
                                       force_dense=(args.suite == "mmlu"
                                                    or args.force_dense))
        if args.force_dense and args.suite != "mmlu":
            arm = arm + "@dense"  # label only — adapter already resolved
        if args.suite == "v1":
            run_v1(model, tokenizer, arm, args.n_samples, log, slog)
        elif args.suite == "v1full":
            run_v1full(model, tokenizer, arm, args.n_samples, log, slog)
        elif args.suite == "govreport":
            run_govreport(model, tokenizer, arm, args.n_samples, log, slog)
        elif args.suite == "mmlu":
            if mmlu_samples is None:
                mmlu_samples = select_mmlu(args.n_samples)
                print(f"[mmlu] selected {len(mmlu_samples)} samples", flush=True)
            # difficulty slot carries the subject -> per-subject accuracies
            run_mcq(model, tokenizer, arm, "mmlu", mmlu_samples, log, slog)
        elif args.suite == "enmc":
            if enmc_samples is None:
                enmc_samples = select_enmc(tokenizer, args.n_samples)
                print(f"[enmc] selected {len(enmc_samples)} samples", flush=True)
                if not enmc_samples:
                    raise SystemExit("no InfiniteBench En.MC samples selected")
            run_mcq(model, tokenizer, arm, "enmc", enmc_samples, log, slog)
        else:
            if v2_samples is None:
                v2_samples = select_v2(tokenizer, args.n_samples)
                print(f"[lb-v2] selected {len(v2_samples)} samples", flush=True)
                if not v2_samples:
                    raise SystemExit("no LongBench-v2 samples fit the token budget")
            run_mcq(model, tokenizer, arm, "v2", v2_samples, log, slog)
        # break the _sample MethodType reference cycle, then drop OUR binding —
        # del must run in this scope or the object survives into the next load
        model._sample = None
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    fh.close()
    sfh.close()
    run.finish()
    print("[longbench] DONE", flush=True)


if __name__ == "__main__":
    main()
