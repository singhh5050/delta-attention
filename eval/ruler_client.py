#!/usr/bin/env python3
"""Thin client around NVIDIA RULER (https://github.com/NVIDIA/RULER) for WP-0.

Vendoring: env/setup.sh clones RULER into third_party/RULER and pins the
commit in third_party/RULER.lock. This module never reimplements RULER —
data comes from RULER's own scripts/data/prepare.py and scoring comes from
RULER's own scripts/eval/synthetic/constants.py metric functions plus the
prediction postprocessing in scripts/eval/evaluate.py. Anything that does
not match those expectations fails loudly with instructions.

Runtime assumptions (asserted, with actionable errors, at call time):
- third_party/RULER exists (run env/setup.sh) and has:
    scripts/data/prepare.py                 (data synthesis entrypoint)
    scripts/data/synthetic.yaml             (task -> base task + args)
    scripts/data/template.py                (Templates dict; we require the
                                             llama-3 chat template, default
                                             key "meta-llama3")
    scripts/eval/synthetic/constants.py     (TASKS dict with 'metric_fn')
    scripts/eval/evaluate.py                (postprocess_pred)
- prepare.py writes {save_dir}/{task}/validation.jsonl with one JSON object
  per line containing at least {"index", "input", "outputs"} (RULER's format).
- Seeds: if prepare.py exposes a --random_seed flag we pass the config seed
  through; otherwise seed 0 means "RULER's own default seeding" and any other
  seed raises (we refuse to pretend a seed we cannot honor).
- The local server (server_hf.py) accepts POST /generate with
  {"text": [...], "sampling_params": {...}} and returns the generated text
  per prompt (dict {"text": [...]}, list of dicts, or list of strings — all
  observed server shapes are handled; anything else raises).
- The tokenizer for length calibration is the HF model itself
  (meta-llama/Llama-3.1-8B-Instruct); HF_TOKEN must grant access.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
RULER_DIR = REPO_ROOT / "third_party" / "RULER"
RULER_LOCK = REPO_ROOT / "third_party" / "RULER.lock"
DATA_CACHE = REPO_ROOT / "data" / "ruler"

MODEL_STR = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TEMPLATE = "meta-llama3"  # RULER's llama-3 chat template key

_PREPARE = RULER_DIR / "scripts" / "data" / "prepare.py"
_SYNTHETIC_YAML = RULER_DIR / "scripts" / "data" / "synthetic.yaml"
_TEMPLATE_PY = RULER_DIR / "scripts" / "data" / "template.py"
_EVAL_CONSTANTS = RULER_DIR / "scripts" / "eval" / "synthetic" / "constants.py"
_EVALUATE_PY = RULER_DIR / "scripts" / "eval" / "evaluate.py"


class RulerSetupError(RuntimeError):
    pass


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise RulerSetupError(
            f"RULER {what} not found at {path}. Run `bash env/setup.sh` to clone and "
            f"pin RULER into {RULER_DIR} (pin recorded in {RULER_LOCK}). If RULER's "
            "layout changed upstream, update eval/ruler_client.py paths — do NOT "
            "reimplement RULER logic."
        )
    return path


def _import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RulerSetupError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# RULER introspection
# ---------------------------------------------------------------------------

def check_template(model_template_type: str = DEFAULT_TEMPLATE) -> str:
    """Assert the requested chat template exists in RULER's template.py."""
    mod = _import_by_path("_ruler_template", _require(_TEMPLATE_PY, "template.py"))
    templates = getattr(mod, "Templates", None)
    if not isinstance(templates, dict):
        raise RulerSetupError(
            f"{_TEMPLATE_PY} has no `Templates` dict — RULER internals changed. "
            "Inspect the file and update check_template()."
        )
    if model_template_type not in templates:
        raise RulerSetupError(
            f"template {model_template_type!r} not in RULER Templates. Available: "
            f"{sorted(templates)}. Pick the Llama-3.1 chat template key and pass it "
            "as model_template_type."
        )
    return model_template_type


def base_task(task: str) -> str:
    """Map a config task name (e.g. niah_single_1) to its RULER base task (niah/qa/...)."""
    import yaml

    with open(_require(_SYNTHETIC_YAML, "synthetic.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if task not in cfg:
        raise RulerSetupError(
            f"task {task!r} not defined in RULER {_SYNTHETIC_YAML}. Available: {sorted(cfg)}")
    entry = cfg[task]
    if not isinstance(entry, dict) or "task" not in entry:
        raise RulerSetupError(
            f"RULER synthetic.yaml entry for {task!r} has no 'task' field: {entry!r}")
    return entry["task"]


def metric_fn_for(task: str):
    """RULER's own metric function for a task (string_match_all / string_match_part)."""
    constants = _import_by_path("_ruler_eval_constants", _require(_EVAL_CONSTANTS, "eval constants"))
    tasks = getattr(constants, "TASKS", None)
    if not isinstance(tasks, dict):
        raise RulerSetupError(f"{_EVAL_CONSTANTS} has no TASKS dict — RULER internals changed.")
    base = base_task(task)
    if base not in tasks or "metric_fn" not in tasks[base]:
        raise RulerSetupError(
            f"RULER eval constants define no metric_fn for base task {base!r} "
            f"(from {task!r}). Available: {sorted(tasks)}.")
    return tasks[base]["metric_fn"]


def postprocess_pred_fn():
    """RULER's own prediction postprocessing (scripts/eval/evaluate.py)."""
    mod = _import_by_path("_ruler_evaluate", _require(_EVALUATE_PY, "evaluate.py"))
    fn = getattr(mod, "postprocess_pred", None)
    if fn is None:
        raise RulerSetupError(
            f"{_EVALUATE_PY} has no postprocess_pred — RULER internals changed. "
            "Point ruler_client.postprocess_pred_fn at the new location; do NOT "
            "reimplement the heuristic."
        )
    return fn


# ---------------------------------------------------------------------------
# data synthesis (RULER's prepare.py) with a local cache
# ---------------------------------------------------------------------------

def _prepare_supports_seed() -> bool:
    return "random_seed" in _require(_PREPARE, "prepare.py").read_text(encoding="utf-8")


def prepare_task_data(
    task: str,
    context_length: int,
    n_samples: int,
    seed: int = 0,
    model_template_type: str = DEFAULT_TEMPLATE,
    tokenizer_path: str = MODEL_STR,
    cache_dir: Path = DATA_CACHE,
) -> List[Dict[str, Any]]:
    """Synthesize (or reuse cached) RULER data; returns the first n_samples samples."""
    check_template(model_template_type)
    save_dir = cache_dir / "synthetic" / str(context_length) / f"seed{seed}_n{n_samples}"
    jsonl = save_dir / task / "validation.jsonl"

    if not jsonl.exists():
        cmd = [
            sys.executable, str(_require(_PREPARE, "prepare.py")),
            "--save_dir", str(save_dir),
            "--benchmark", "synthetic",
            "--task", task,
            "--tokenizer_path", tokenizer_path,
            "--tokenizer_type", "hf",
            "--max_seq_length", str(context_length),
            "--model_template_type", model_template_type,
            "--num_samples", str(n_samples),
        ]
        if _prepare_supports_seed():
            cmd += ["--random_seed", str(seed)]
        elif seed != 0:
            raise RulerSetupError(
                f"config seed={seed} but this RULER pin's prepare.py exposes no "
                "--random_seed flag; refusing to fake determinism. Use seed 0 "
                "(RULER's own default seeding) or extend the pin."
            )
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ruler_client] preparing data: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(_PREPARE.parent), capture_output=True, text=True)
        if proc.returncode != 0 or not jsonl.exists():
            raise RulerSetupError(
                f"RULER prepare.py failed for task={task} ctx={context_length} "
                f"(rc={proc.returncode}); expected {jsonl}.\n"
                f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}\n"
                "Common causes: missing task source data (run env/setup.sh, which runs "
                "RULER's essay/QA download scripts), missing HF_TOKEN for the tokenizer, "
                "or missing RULER python deps (wonderwords, html2text, nltk, tenacity)."
            )

    samples: List[Dict[str, Any]] = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if len(samples) < n_samples:
        raise RulerSetupError(
            f"{jsonl} has only {len(samples)} samples, needed {n_samples}. "
            "Delete the cache dir and re-run, or check prepare.py output.")
    samples = samples[:n_samples]
    for s in samples:
        for key in ("index", "input", "outputs"):
            if key not in s:
                raise RulerSetupError(
                    f"RULER sample missing key {key!r} in {jsonl} — format changed; "
                    f"sample keys: {sorted(s)}")
    return samples


# ---------------------------------------------------------------------------
# generation against the local server + scoring with RULER's own logic
# ---------------------------------------------------------------------------

def _extract_texts(payload: Any) -> List[str]:
    if isinstance(payload, dict) and "text" in payload:
        payload = payload["text"]
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list):
        out = []
        for item in payload:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and "text" in item:
                out.append(item["text"])
            else:
                raise RuntimeError(f"unexpected /generate response item: {item!r}")
        return out
    raise RuntimeError(f"unexpected /generate response shape: {type(payload)}: {payload!r}")


def generate_one(server_url: str, prompt: str, max_new_tokens: int,
                 timeout_s: float = 900.0) -> str:
    """One prompt -> one completion via the local server. Greedy (temperature 0)."""
    import requests

    r = requests.post(
        f"{server_url.rstrip('/')}/generate",
        json={"text": prompt,
              "sampling_params": {"max_new_tokens": max_new_tokens,
                                  "temperature": 0.0, "top_p": 1.0, "top_k": 1,
                                  "stop": []}},
        timeout=timeout_s,
    )
    r.raise_for_status()
    texts = _extract_texts(r.json())
    if len(texts) != 1:
        raise RuntimeError(f"expected 1 completion, got {len(texts)}")
    return texts[0]


def run_samples(server_url: str, samples: List[Dict[str, Any]], max_new_tokens: int,
                timeout_s: float = 900.0) -> Dict[str, Any]:
    """Send samples sequentially to one server. No silent retries: failures counted.

    Returns {"predictions": {index: text}, "latencies": [s,...], "failures": int}.
    """
    predictions: Dict[Any, str] = {}
    latencies: List[float] = []
    failures = 0
    for s in samples:
        t0 = time.monotonic()
        try:
            predictions[s["index"]] = generate_one(server_url, s["input"],
                                                   max_new_tokens, timeout_s)
            latencies.append(time.monotonic() - t0)
        except Exception as e:  # noqa: BLE001 — counted, surfaced, never retried
            failures += 1
            print(f"[ruler_client] sample {s['index']} FAILED on {server_url}: {e!r}")
    return {"predictions": predictions, "latencies": latencies, "failures": failures}


def score_task(task: str, predictions: Dict[Any, str],
               samples: List[Dict[str, Any]]) -> float:
    """Accuracy in [0, 100] using RULER's own metric_fn + postprocess_pred."""
    if not predictions:
        raise RuntimeError(f"no predictions to score for task {task!r} (all samples failed?)")
    post = postprocess_pred_fn()
    metric = metric_fn_for(task)
    preds, refs = [], []
    for s in samples:
        if s["index"] in predictions:
            preds.append(post(predictions[s["index"]]))
            refs.append(s["outputs"])
    score = metric(preds, refs)
    return float(score)


def run_task(
    server_url: str,
    task: str,
    context_length: int,
    n_samples: int,
    seed: int = 0,
    max_new_tokens: int = 128,
    model_template_type: str = DEFAULT_TEMPLATE,
) -> Dict[str, Any]:
    """Single-server convenience wrapper: prepare -> generate -> score.

    Returns {"accuracy": float 0-100, "n": int scored, "latencies": [s], "failures": int}.
    run_matrix.py uses the lower-level pieces directly to shard samples across
    multiple servers; Gate 2 and standalone use go through this.
    """
    samples = prepare_task_data(task, context_length, n_samples, seed=seed,
                                model_template_type=model_template_type)
    res = run_samples(server_url, samples, max_new_tokens)
    accuracy = score_task(task, res["predictions"], samples)
    return {"accuracy": accuracy, "n": len(res["predictions"]),
            "latencies": res["latencies"], "failures": res["failures"]}
