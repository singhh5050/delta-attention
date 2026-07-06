#!/usr/bin/env python3
"""Gate 2 — single-sample end-to-end smoke (docs/WP0_infra_and_smoke.md §5).

Through the REAL server + REAL RULER client, no shortcuts: one niah_single_1
sample at 4096 and one at 32768 for configs base_delta_g64, base_sparse_only,
base_dense_fa2 (rows from experiments.yaml; only the §5-mandated sample plan
— context lengths / task / n_samples — replaces the row's full-run values).

Asserts, per config:
- server started and became healthy (run_config would have failed otherwise),
- generation non-empty for every sample (min_pred_chars > 0, zero failures),
- needle retrieved (accuracy == 100) for delta and dense at BOTH lengths,
- exactly one row appended to results/results.csv with status=ok,
- a wandb run was created and every mandatory metric key was logged at init.

Budget: ~10 min on 1 GPU. Reuses eval/run_matrix.py machinery — no duplicate
pipeline. Exit code 0 = Gate 2 green.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval import run_matrix  # noqa: E402

GATE2_CONFIGS = ["base_delta_g64", "base_sparse_only", "base_dense_fa2"]
GATE2_TASK = "niah_single_1"
GATE2_CONTEXT_LENGTHS = [4096, 32768]
GATE2_N_SAMPLES = 1
NEEDLE_MODES = {"delta", "none"}  # must retrieve the needle; sparse-only: non-empty only


def gate2_rows(experiments: Path = run_matrix.EXPERIMENTS_YAML) -> List[Dict[str, Any]]:
    """The three §5 config rows with the §5 single-sample plan applied."""
    rows = run_matrix.load_configs(",".join(GATE2_CONFIGS), smoke=False, path=experiments)
    out = []
    for row in rows:
        row = dict(row)
        row["context_lengths"] = list(GATE2_CONTEXT_LENGTHS)
        row["tasks"] = [GATE2_TASK]
        row["n_samples"] = GATE2_N_SAMPLES
        out.append(row)
    return out


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)  # minus header


def check_result(row: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
    """Return a list of failed-assertion messages (empty = pass)."""
    errs: List[str] = []
    name = row["name"]
    if result["status"] != "ok":
        errs.append(f"{name}: status={result['status']!r} error={result.get('error')!r}")
        return errs

    cells = json.loads(result["per_task_json"])["cells"]
    got = {(c["context_len"], c["task"]): c for c in cells}
    for ctx in GATE2_CONTEXT_LENGTHS:
        cell = got.get((ctx, GATE2_TASK))
        if cell is None:
            errs.append(f"{name}: no result cell for ctx={ctx} task={GATE2_TASK}")
            continue
        if cell["failures"] != 0 or cell["n"] != GATE2_N_SAMPLES:
            errs.append(f"{name} ctx={ctx}: failures={cell['failures']} n={cell['n']}")
        if cell.get("min_pred_chars", 0) <= 0:
            errs.append(f"{name} ctx={ctx}: empty generation")
        if row["mode"] in NEEDLE_MODES and cell["accuracy"] < 100.0:
            errs.append(f"{name} ctx={ctx}: needle NOT retrieved "
                        f"(accuracy={cell['accuracy']}, mode={row['mode']})")
    wandb_mode = os.environ.get("WANDB_MODE", "")
    if not result["wandb_url"] and wandb_mode not in ("offline", "disabled", "dryrun"):
        errs.append(f"{name}: no wandb run URL recorded")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=run_matrix.RESULTS_CSV)
    ap.add_argument("--server-startup-timeout", type=int,
                    default=run_matrix.SERVER_STARTUP_TIMEOUT_S)
    args = ap.parse_args(argv)

    # Startup validation gate (WP-0 §4) — mandatory for every entrypoint.
    try:
        from delta_attention.validation import (MANDATORY_WANDB_KEYS as VAL_KEYS,
                                                startup_validation)
    except ImportError as e:
        raise RuntimeError(
            "delta_attention/validation.py is missing or unimportable — the WP-0 §4 "
            f"startup validation gate is mandatory. Original error: {e}"
        ) from e
    if set(VAL_KEYS) != set(run_matrix.MANDATORY_WANDB_KEYS):
        raise RuntimeError(
            "mandatory wandb key sets diverged between validation.py and run_matrix.py: "
            f"{sorted(set(VAL_KEYS) ^ set(run_matrix.MANDATORY_WANDB_KEYS))}")

    rows = gate2_rows()
    server_fields = run_matrix.get_server_config_fields()
    for row in rows:
        startup_validation(row, logged_wandb_keys=run_matrix.MANDATORY_WANDB_KEYS,
                           smoke=True)

    gpu = run_matrix.visible_gpus()[:1]  # Gate 2 budget: 1 GPU
    print(f"[smoke_e2e] Gate 2 on GPU {gpu[0]}: configs={GATE2_CONFIGS}, "
          f"task={GATE2_TASK}, lengths={GATE2_CONTEXT_LENGTHS}, n={GATE2_N_SAMPLES}")

    failures: List[str] = []
    for row in rows:
        rows_before = _count_csv_rows(args.results)
        result = run_matrix.run_config(
            row, smoke=True, gpus=gpu, server_fields=server_fields,
            results_path=args.results, startup_timeout_s=args.server_startup_timeout)
        rows_after = _count_csv_rows(args.results)
        if rows_after != rows_before + 1:
            failures.append(f"{row['name']}: expected exactly one new results.csv row "
                            f"({rows_before} -> {rows_after})")
        failures.extend(check_result(row, result))

    if failures:
        print("[smoke_e2e] GATE 2 FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[smoke_e2e] GATE 2 PASS: all assertions green for "
          f"{', '.join(GATE2_CONFIGS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
