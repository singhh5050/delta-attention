#!/usr/bin/env python3
"""WP-0 experiment runner (docs/WP0_infra_and_smoke.md §2).

Executes rows of experiments.yaml verbatim:

    python eval/run_matrix.py --configs base_delta_g64,base_dense_fa2
    python eval/run_matrix.py --configs night1_all --smoke     # Gate 3
    python eval/run_matrix.py --configs all

Per config: start one server_hf.py per visible GPU, wait for /healthz
(startup healthcheck only — never monitors running jobs), run the RULER
tasks through eval/ruler_client.py with samples sharded across servers,
tear the servers down, append EXACTLY ONE row to results/results.csv and
log one wandb run with the mandatory metric set from the master plan.

Failed configs are never retried silently: status=failed + error string is
recorded and the runner continues. Configs that need not-yet-implemented
server features (decode_mode != dense, stride_policy != fixed, checkpoint)
are recorded with status=unsupported and skipped with a clear message.

The runner introspects delta_attention.config.Config to decide which YAML
keys become server CLI flags; everything else is recorded (results.csv
`per_task_json`/wandb config) — never silently dropped.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as _dt
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_YAML = REPO_ROOT / "experiments.yaml"
RESULTS_CSV = REPO_ROOT / "results" / "results.csv"
SERVER_LOG_DIR = REPO_ROOT / "results" / "server_logs"

# Fixed by experiments.yaml header comment ("All configs: model=..., seed=0,
# temperature=0, max_new_tokens=128"). Values, not discretion.
MODEL_STR = "meta-llama/Llama-3.1-8B-Instruct"
TEMPERATURE = 0.0
IMPLICIT_ROW_DEFAULTS: Dict[str, Any] = {"seed": 0, "max_new_tokens": 128}

# YAML keys that describe the *harness* run, never server flags.
HARNESS_ONLY_KEYS = {"name", "context_lengths", "tasks", "n_samples", "expected_anchor"}

# results.csv schema — documented in results/SCHEMA.md. One row per config run.
CSV_COLUMNS = [
    "config_name", "status", "mode", "attn_implementation", "gamma",
    "sliding_window", "decode_mode", "gamma_dec", "refresh_policy",
    "stride_policy", "context_len", "tasks", "n_samples", "accuracy",
    "per_task_json", "samples_per_sec", "prefill_ms_p50",
    "decode_ms_per_token_p50", "oom_count", "effective_sparsity",
    "wall_time_s", "wandb_url", "error", "git_sha", "timestamp",
]

MANDATORY_WANDB_KEYS = (
    "config_name", "mode", "gamma", "gamma_dec", "refresh_policy",
    "context_len", "task", "n_samples", "accuracy", "samples_per_sec",
    "prefill_ms_p50", "decode_ms_per_token_p50", "oom_count",
)

SERVER_STARTUP_TIMEOUT_S = 600  # model load can take minutes; startup-only healthcheck


# ===========================================================================
# YAML loading / config resolution (offline-testable, no GPU/network)
# ===========================================================================

def load_experiments(path: Path = EXPERIMENTS_YAML) -> Dict[str, Any]:
    """Load experiments.yaml and verify YAML merge keys actually resolved.

    PyYAML's SafeLoader supports `<<:` merge keys natively; we still verify
    because a loader swap that drops them would silently break every config.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    for section in ("defaults", "smoke_overrides", "configs", "groups"):
        if section not in doc:
            raise RuntimeError(f"{path}: missing section {section!r}")
    names = [c.get("name") for c in doc["configs"]]
    if len(names) != len(set(names)):
        raise RuntimeError(f"{path}: duplicate config names")
    must_merge = {"attn_implementation", "sliding_window", "mode", "delta_lambda",
                  "decode_mode", "stride_policy", "context_lengths", "tasks",
                  "n_samples", "log_drift"}
    for cfg in doc["configs"]:
        missing = must_merge - set(cfg)
        if missing:
            raise RuntimeError(
                f"{path}: config {cfg.get('name')!r} is missing {sorted(missing)} — "
                "YAML merge keys (<<: *defaults) did not resolve. Use a loader "
                "that supports merge keys (PyYAML safe_load does)."
            )
    return doc


def resolve_config_names(doc: Dict[str, Any], configs_arg: str) -> List[str]:
    """Expand --configs (comma list of config names and/or group names, or 'all')."""
    all_names = [c["name"] for c in doc["configs"]]
    if configs_arg.strip() == "all":
        return list(all_names)
    groups = doc.get("groups", {})
    out: List[str] = []
    for token in [t.strip() for t in configs_arg.split(",") if t.strip()]:
        if token in groups:
            expansion = groups[token]
        elif token in all_names:
            expansion = [token]
        else:
            raise SystemExit(
                f"--configs: {token!r} is neither a config nor a group. "
                f"Configs: {all_names}. Groups: {list(groups)}."
            )
        for name in expansion:
            if name not in all_names:
                raise SystemExit(f"group expands to unknown config {name!r}")
            if name not in out:
                out.append(name)
    if not out:
        raise SystemExit("--configs resolved to an empty set")
    return out


def apply_smoke_overrides(row: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply smoke_overrides verbatim. Only context_lengths/tasks/n_samples may change."""
    allowed = {"context_lengths", "tasks", "n_samples"}
    extra = set(overrides) - allowed
    if extra:
        raise RuntimeError(f"smoke_overrides may only touch {sorted(allowed)}, found {sorted(extra)}")
    out = dict(row)
    out.update(overrides)
    return out


def load_configs(configs_arg: str, smoke: bool, path: Path = EXPERIMENTS_YAML) -> List[Dict[str, Any]]:
    """Resolved, merged config rows in execution order (never mutates the YAML)."""
    doc = load_experiments(path)
    by_name = {c["name"]: c for c in doc["configs"]}
    rows = []
    for name in resolve_config_names(doc, configs_arg):
        row = dict(by_name[name])
        for k, v in IMPLICIT_ROW_DEFAULTS.items():
            row.setdefault(k, v)
        if smoke:
            row = apply_smoke_overrides(row, doc["smoke_overrides"])
        rows.append(row)
    return rows


# ===========================================================================
# Server Config introspection and flag construction
# ===========================================================================

def get_server_config_fields() -> frozenset:
    """Field names of the delta_attention.config.Config dataclass.

    Prefers importing the real dataclass. If the import fails because heavy
    deps (hip_attn/torch) are absent — e.g. on a dev box — falls back to
    statically parsing config.py with ast (prints a note; not a silent
    fallback, and the parsed file is the same source of truth).
    """
    try:
        import dataclasses

        from delta_attention.config import Config  # noqa: PLC0415
        return frozenset(f.name for f in dataclasses.fields(Config))
    except ImportError as e:
        print(f"[run_matrix] note: importing delta_attention.config failed ({e}); "
              "statically parsing config.py for Config fields.")
        return _parse_config_fields(REPO_ROOT / "delta_attention" / "config.py")


def _parse_config_fields(config_py: Path) -> frozenset:
    tree = ast.parse(config_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            fields = [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
            if not fields:
                raise RuntimeError(f"{config_py}: Config class has no annotated fields")
            return frozenset(fields)
    raise RuntimeError(f"{config_py}: no Config class found")


def unsupported_reason(row: Dict[str, Any], server_fields: frozenset) -> Optional[str]:
    """Reason string if the config needs a feature the server Config lacks.

    WP-1/2/3 extend Config; once a field lands in Config the config becomes
    runnable automatically (introspection, not a hardcoded list of names).
    """
    reasons = []
    if row.get("decode_mode", "dense") != "dense" and "decode_mode" not in server_fields:
        reasons.append(f"decode_mode={row['decode_mode']!r} needs WP-3 (Config has no decode_mode)")
    if row.get("stride_policy", "fixed") != "fixed" and "stride_policy" not in server_fields:
        reasons.append(f"stride_policy={row['stride_policy']!r} needs WP-1 (Config has no stride_policy)")
    if row.get("checkpoint") and "checkpoint" not in server_fields:
        reasons.append(f"checkpoint={row['checkpoint']!r} needs WP-2 (Config has no checkpoint)")
    return "; ".join(reasons) or None


def build_server_cmd(
    row: Dict[str, Any], port: int, server_fields: frozenset
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any], List[str]]:
    """CLI for server_hf.py for one config row.

    Returns (cmd, passed_flags, unknown_to_server, notes).
    - Only YAML keys that exist as Config dataclass fields become flags.
    - mode "none" (dense FA2 baseline): server_hf.main asserts mode ∈
      {delta, recompute, sparse-only} but ignores mode entirely when
      attn_implementation == flash_attention_2, so we pass --mode delta and
      print a note. Any other attn_implementation with mode "none" is an error.
    - Unknown-to-server keys are returned for recording, never dropped.
    """
    notes: List[str] = []
    passed: Dict[str, Any] = {
        "port": port,
        "model_str": MODEL_STR,
        "temperature": TEMPERATURE,
        "max_new_tokens": row.get("max_new_tokens", IMPLICIT_ROW_DEFAULTS["max_new_tokens"]),
    }
    unknown: Dict[str, Any] = {}

    for key, val in row.items():
        if key in HARNESS_ONLY_KEYS or key in passed:
            continue
        if key in server_fields:
            passed[key] = val
        else:
            unknown[key] = val

    if passed.get("mode") == "none":
        if row["attn_implementation"] != "flash_attention_2":
            raise RuntimeError(
                f"config {row['name']!r}: mode 'none' is only valid with "
                f"attn_implementation flash_attention_2, got {row['attn_implementation']!r}"
            )
        passed["mode"] = "delta"
        notes.append(
            f"config {row['name']!r}: mode 'none' mapped to '--mode delta' — the "
            "flash_attention_2 path in server_hf.py ignores mode entirely."
        )

    for f in ("port", "max_new_tokens", "model_str", "temperature"):
        if f not in server_fields:
            raise RuntimeError(
                f"server Config no longer has field {f!r}; update build_server_cmd. "
                f"Known fields: {sorted(server_fields)}"
            )

    cmd = [sys.executable, str(REPO_ROOT / "server_hf.py")]
    for key, val in passed.items():
        # argparse_dataclass exposes fields as --dashed-names (see README usage
        # block). Bools are presence-only store_true flags: emit the flag for
        # True, nothing for False ("--log-drift True" is a parse error).
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd += [flag, str(val)]
    return cmd, passed, unknown, notes


# ===========================================================================
# Server lifecycle
# ===========================================================================

class ServerHandle:
    def __init__(self, proc: subprocess.Popen, port: int, gpu: str, log_path: Path,
                 drift_path: Optional[Path] = None):
        self.proc = proc
        self.port = port
        self.gpu = gpu
        self.log_path = log_path
        self.drift_path = drift_path
        self.url = f"http://127.0.0.1:{port}"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def visible_gpus() -> List[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        gpus = [g.strip() for g in env.split(",") if g.strip()]
        if gpus:
            return gpus
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        gpus = [str(i) for i, line in enumerate(out.stdout.splitlines()) if line.startswith("GPU")]
        if gpus:
            return gpus
    except (OSError, subprocess.CalledProcessError):
        pass
    raise SystemExit(
        "No visible GPUs (CUDA_VISIBLE_DEVICES unset and nvidia-smi found none). "
        "run_matrix needs at least one GPU."
    )


def start_server(row: Dict[str, Any], gpu: str, server_fields: frozenset,
                 log_dir: Path = SERVER_LOG_DIR) -> Tuple[ServerHandle, Dict[str, Any], List[str]]:
    port = find_free_port()
    cmd, passed, unknown, notes = build_server_cmd(row, port, server_fields)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{row['name']}_gpu{gpu}_port{port}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    drift_path = None
    if row.get("log_drift"):
        drift_path = log_dir / f"{row['name']}_gpu{gpu}_port{port}_drift.jsonl"
        drift_path.unlink(missing_ok=True)
        env["DELTA_DRIFT_LOG"] = str(drift_path)
    logf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env=env, cwd=str(REPO_ROOT))
    for n in notes:
        print(f"[run_matrix] NOTE: {n}")
    print(f"[run_matrix] started server for {row['name']!r} on GPU {gpu} port {port} "
          f"(pid {proc.pid}, log {log_path})")
    return ServerHandle(proc, port, gpu, log_path, drift_path), unknown, notes


def wait_healthy(handle: ServerHandle, timeout_s: int = SERVER_STARTUP_TIMEOUT_S) -> None:
    """Poll GET /healthz until 200. Startup healthcheck ONLY (allowed by the plan)."""
    import requests

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if handle.proc.poll() is not None:
            raise RuntimeError(
                f"server on GPU {handle.gpu} exited with code {handle.proc.returncode} "
                f"during startup. Log tail:\n{_log_tail(handle.log_path)}"
            )
        try:
            r = requests.get(f"{handle.url}/healthz", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2.0)
    raise RuntimeError(
        f"server on GPU {handle.gpu} not healthy after {timeout_s}s. "
        f"Log tail:\n{_log_tail(handle.log_path)}"
    )


def stop_server(handle: ServerHandle) -> None:
    if handle.proc.poll() is None:
        handle.proc.send_signal(signal.SIGTERM)
        try:
            handle.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            handle.proc.kill()
            handle.proc.wait(timeout=20)


def _log_tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return "<log unreadable>"


def log_drift_telemetry(run, handles: Sequence[ServerHandle]) -> Optional[Dict[str, float]]:
    """Ingest DELTA_DRIFT_LOG sidecars and log per-layer delta_interanchor_cos
    histograms + means to wandb. Returns {layer: mean} or None if no telemetry."""
    from delta_attention.drift import aggregate_drift, hist_bin_edges

    lines: List[Dict[str, Any]] = []
    for h in handles:
        if h.drift_path is None or not h.drift_path.exists():
            continue
        with open(h.drift_path, "r", encoding="utf-8") as f:
            lines += [json.loads(ln) for ln in f if ln.strip()]
    if not lines:
        return None
    agg = aggregate_drift(lines)
    edges = hist_bin_edges()
    payload: Dict[str, Any] = {}
    try:
        import wandb  # noqa: PLC0415
        for layer, a in sorted(agg.items()):
            payload[f"delta_interanchor_cos/hist_layer_{layer:02d}"] = wandb.Histogram(
                np_histogram=(a["hist"], edges))
    except ImportError:
        pass  # wandb_init already enforced wandb presence; keep means regardless
    means = {layer: a["mean"] for layer, a in sorted(agg.items())}
    for layer, mean in means.items():
        payload[f"delta_interanchor_cos/mean_layer_{layer:02d}"] = mean
    run.log(payload)
    print(f"[run_matrix] drift telemetry: {sum(a['requests'] for a in agg.values())} "
          f"layer-records over {len(agg)} layers; per-layer mean cos "
          f"min={min(means.values()):.4f} max={max(means.values()):.4f}")
    return means


def count_ooms(handles: Sequence[ServerHandle]) -> int:
    n = 0
    for h in handles:
        try:
            n += h.log_path.read_text(encoding="utf-8", errors="replace").count(
                "CUDA out of memory")
        except OSError:
            pass
    return n


# ===========================================================================
# results.csv
# ===========================================================================

def ensure_results_csv(path: Path = RESULTS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)


def append_result_row(row: Dict[str, Any], path: Path = RESULTS_CSV) -> None:
    ensure_results_csv(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(
            {k: ("" if row.get(k) is None else row.get(k, "")) for k in CSV_COLUMNS})


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def base_result_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config_name": row["name"],
        "status": "",
        "mode": row["mode"],
        "attn_implementation": row["attn_implementation"],
        "gamma": row.get("delta_lambda"),
        "sliding_window": row.get("sliding_window"),
        "decode_mode": row.get("decode_mode"),
        "gamma_dec": row.get("gamma_dec"),
        "refresh_policy": row.get("refresh_policy"),
        "stride_policy": row.get("stride_policy"),
        "context_len": json.dumps(row["context_lengths"]),
        "tasks": json.dumps(row["tasks"]),
        "n_samples": row["n_samples"],
        "accuracy": None,
        "per_task_json": "",
        "samples_per_sec": None,
        "prefill_ms_p50": None,          # not separable through HTTP yet; null
        "decode_ms_per_token_p50": None,  # WP-3 instrumentation; null
        "oom_count": 0,
        "effective_sparsity": None,       # WP-1 instrumentation; null
        "wall_time_s": None,
        "wandb_url": "",
        "error": "",
        "git_sha": git_sha(),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


# ===========================================================================
# wandb
# ===========================================================================

def wandb_init(row: Dict[str, Any], smoke: bool, extra_config: Dict[str, Any]):
    """Init a wandb run and log EVERY mandatory key once (zeros/nulls)."""
    try:
        import wandb  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "wandb is required (master plan §mandatory metrics). "
            "pip install wandb, or export WANDB_MODE=offline for air-gapped runs."
        ) from e
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if not project and os.environ.get("WANDB_MODE") not in ("offline", "disabled", "dryrun"):
        raise RuntimeError("WANDB_PROJECT env var is required (WANDB_ENTITY too, unless "
                           "your default entity applies). Or set WANDB_MODE=offline.")
    run = wandb.init(
        entity=entity, project=project, name=row["name"],
        config={**row, "smoke": smoke, "temperature": TEMPERATURE,
                "model_str": MODEL_STR, **extra_config},
        reinit=True,
    )
    init_payload = {k: 0 for k in MANDATORY_WANDB_KEYS}
    init_payload.update(config_name=row["name"], mode=row["mode"],
                        gamma=row.get("delta_lambda", 0),
                        gamma_dec=row.get("gamma_dec", 0) or 0,
                        refresh_policy=row.get("refresh_policy") or "none",
                        task="__init__")
    run.log(init_payload)
    return run


# ===========================================================================
# per-config execution
# ===========================================================================

def _percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    vs = sorted(vals)
    i = min(int(round(p / 100.0 * (len(vs) - 1))), len(vs) - 1)
    return vs[i]


def run_config(row: Dict[str, Any], smoke: bool, gpus: List[str],
               server_fields: frozenset, results_path: Path = RESULTS_CSV,
               startup_timeout_s: int = SERVER_STARTUP_TIMEOUT_S) -> Dict[str, Any]:
    """Execute one config end to end and append exactly one results.csv row."""
    from eval import ruler_client  # local import: needs requests + RULER at runtime

    t0 = time.monotonic()
    result = base_result_row(row)
    _, _, unknown, notes = build_server_cmd(row, 0, server_fields)
    result_extra = {"unknown_to_server": unknown, "server_notes": notes}

    try:
        run = wandb_init(row, smoke, result_extra)
    except Exception as e:  # noqa: BLE001 — record status=failed, continue to next config
        err = f"wandb init failed: {type(e).__name__}: {e}"
        print(f"[run_matrix] FAILED {row['name']!r}: {err}")
        result.update(status="failed", error=err,
                      wall_time_s=round(time.monotonic() - t0, 2))
        append_result_row(result, results_path)
        return result
    result["wandb_url"] = getattr(run, "url", "") or ""

    reason = unsupported_reason(row, server_fields)
    if reason:
        print(f"[run_matrix] SKIP {row['name']!r}: unsupported — {reason}")
        result.update(status="unsupported", error=reason,
                      wall_time_s=round(time.monotonic() - t0, 2))
        run.summary["status"] = "unsupported"
        run.finish()
        append_result_row(result, results_path)
        return result

    handles: List[ServerHandle] = []
    per_task: List[Dict[str, Any]] = []
    all_latencies: List[float] = []
    total_done = 0
    eval_seconds = 0.0
    try:
        for gpu in gpus:
            h, _, _ = start_server(row, gpu, server_fields)
            handles.append(h)
        for h in handles:
            wait_healthy(h, timeout_s=startup_timeout_s)
        print(f"[run_matrix] {len(handles)} server(s) healthy for {row['name']!r}")

        for ctx in row["context_lengths"]:
            for task in row["tasks"]:
                samples = ruler_client.prepare_task_data(
                    task=task, context_length=ctx, n_samples=row["n_samples"],
                    seed=row.get("seed", 0),
                )
                shards = [samples[i::len(handles)] for i in range(len(handles))]
                t_cell = time.monotonic()
                with ThreadPoolExecutor(max_workers=len(handles)) as ex:
                    futs = [
                        ex.submit(ruler_client.run_samples, h.url, shard,
                                  row.get("max_new_tokens", 128))
                        for h, shard in zip(handles, shards) if shard
                    ]
                    shard_results = [f.result() for f in futs]
                cell_secs = time.monotonic() - t_cell
                preds = {}
                latencies: List[float] = []
                failures = 0
                for sr in shard_results:
                    preds.update(sr["predictions"])
                    latencies.extend(sr["latencies"])
                    failures += sr["failures"]
                acc = ruler_client.score_task(task, preds, samples)
                n_ok = len(preds)
                per_task.append({"context_len": ctx, "task": task, "n": n_ok,
                                 "failures": failures, "accuracy": acc,
                                 "min_pred_chars": min(
                                     (len(p.strip()) for p in preds.values()), default=0),
                                 "seconds": round(cell_secs, 2)})
                all_latencies.extend(latencies)
                total_done += n_ok
                eval_seconds += cell_secs
                run.log({
                    "config_name": row["name"], "mode": row["mode"],
                    "gamma": row.get("delta_lambda", 0),
                    "gamma_dec": row.get("gamma_dec", 0) or 0,
                    "refresh_policy": row.get("refresh_policy") or "none",
                    "context_len": ctx, "task": task, "n_samples": n_ok,
                    "accuracy": acc, "samples_per_sec": n_ok / max(cell_secs, 1e-9),
                    "prefill_ms_p50": 0, "decode_ms_per_token_p50": 0,
                    "oom_count": count_ooms(handles),
                    "request_latency_s_p50": _percentile(latencies, 50),
                })
                print(f"[run_matrix] {row['name']} ctx={ctx} task={task}: "
                      f"acc={acc:.2f} n={n_ok} failures={failures} ({cell_secs:.1f}s)")

        drift_means = log_drift_telemetry(run, handles)
        if drift_means is not None:
            result_extra["drift_mean_by_layer"] = drift_means

        accs = [c["accuracy"] for c in per_task]
        result.update(
            status="ok",
            accuracy=round(sum(accs) / len(accs), 4) if accs else None,
            per_task_json=json.dumps({"cells": per_task, **result_extra}),
            samples_per_sec=round(total_done / max(eval_seconds, 1e-9), 4),
            oom_count=count_ooms(handles),
        )
        try:
            import wandb  # noqa: PLC0415
            run.log({"per_task_accuracy": wandb.Table(
                columns=["context_len", "task", "n", "failures", "accuracy"],
                data=[[c["context_len"], c["task"], c["n"], c["failures"], c["accuracy"]]
                      for c in per_task])})
        except Exception as e:  # noqa: BLE001 — table failure must not lose the row
            print(f"[run_matrix] WARN: wandb table log failed: {e!r}")
        run.summary["status"] = "ok"
        run.summary["accuracy"] = result["accuracy"]
    except Exception as e:  # noqa: BLE001 — record and continue, never retry silently
        err = f"{type(e).__name__}: {e}"
        print(f"[run_matrix] FAILED {row['name']!r}: {err}")
        result.update(status="failed", error=err,
                      oom_count=count_ooms(handles),
                      per_task_json=json.dumps({"cells": per_task, **result_extra}))
        run.summary["status"] = "failed"
    finally:
        for h in handles:
            stop_server(h)
    result["wall_time_s"] = round(time.monotonic() - t0, 2)
    run.log({"wall_time_s": result["wall_time_s"]})
    run.finish()
    append_result_row(result, results_path)
    return result


# ===========================================================================
# reporting
# ===========================================================================

def format_config_table(rows: List[Dict[str, Any]], server_fields: frozenset) -> str:
    hdr = ["name", "mode", "attn", "gamma", "decode_mode", "stride", "ctx_lengths",
           "n_tasks", "n_samples", "est_samples", "supported"]
    lines = ["\t".join(hdr)]
    total = 0
    for r in rows:
        est = len(r["context_lengths"]) * len(r["tasks"]) * r["n_samples"]
        reason = unsupported_reason(r, server_fields)
        if not reason:
            total += est
        lines.append("\t".join(str(x) for x in [
            r["name"], r["mode"], r["attn_implementation"], r.get("delta_lambda"),
            r.get("decode_mode"), r.get("stride_policy"),
            ",".join(map(str, r["context_lengths"])), len(r["tasks"]), r["n_samples"],
            est, "no: " + reason if reason else "yes"]))
    lines.append(f"estimated total samples (supported configs): {total}")
    return "\n".join(lines)


def format_summary(results: List[Dict[str, Any]]) -> str:
    hdr = ["config", "status", "accuracy", "samples/s", "wall_time_s", "error"]
    lines = ["\t".join(hdr)]
    for r in results:
        lines.append("\t".join(str(x) for x in [
            r["config_name"], r["status"], r.get("accuracy"),
            r.get("samples_per_sec"), r.get("wall_time_s"),
            (r.get("error") or "")[:120]]))
    return "\n".join(lines)


# ===========================================================================
# entrypoint
# ===========================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True,
                    help="comma list of config and/or group names from experiments.yaml, or 'all'")
    ap.add_argument("--smoke", action="store_true",
                    help="apply smoke_overrides from experiments.yaml (same code path)")
    ap.add_argument("--experiments", type=Path, default=EXPERIMENTS_YAML)
    ap.add_argument("--results", type=Path, default=RESULTS_CSV)
    ap.add_argument("--server-startup-timeout", type=int, default=SERVER_STARTUP_TIMEOUT_S)
    args = ap.parse_args(argv)

    # Startup validation gate (WP-0 §4) — written in parallel; a missing module
    # is an operator-visible error, never a silent skip.
    try:
        from delta_attention.validation import startup_validation
    except ImportError as e:
        raise RuntimeError(
            "delta_attention/validation.py is missing or unimportable — the WP-0 §4 "
            "startup validation gate is mandatory for every entrypoint. "
            f"Original error: {e}"
        ) from e

    rows = load_configs(args.configs, args.smoke, args.experiments)
    server_fields = get_server_config_fields()

    print("[run_matrix] resolved config table:")
    print(format_config_table(rows, server_fields))

    for row in rows:
        startup_validation(row, logged_wandb_keys=MANDATORY_WANDB_KEYS, smoke=args.smoke)

    gpus = visible_gpus()
    print(f"[run_matrix] visible GPUs: {gpus}")
    ensure_results_csv(args.results)

    results = []
    for row in rows:
        results.append(run_config(row, args.smoke, gpus, server_fields,
                                   results_path=args.results,
                                   startup_timeout_s=args.server_startup_timeout))

    print("[run_matrix] summary:")
    print(format_summary(results))
    print(f"[run_matrix] results appended to {args.results}")
    return 0 if all(r["status"] in ("ok", "unsupported") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
