#!/usr/bin/env python3
"""Gate 3 post-run assertions (docs/WP0_infra_and_smoke.md §6).

Run AFTER:  python eval/run_matrix.py --configs night1_all --smoke

Checks the latest results/results.csv row per night1_all config:
1. Every night1_all config has a results row. Rows must be status=ok —
   except configs the runner declared status=unsupported because they need
   WP-1/WP-3 server features that have not landed yet (decode_mode != dense,
   stride_policy != fixed). Those are reported prominently and tolerated
   unless --require-all-ok is passed (once WP-1/3 land, run with
   --require-all-ok; the spec-literal gate is "all ok").
2. Ordering sanity at 4K: acc(dense) >= acc(delta) >= acc(sparse_only) - 5pt.
3. base_delta_g64 / base_sparse_only within +/-10pt of the paper 4K anchors
   (96.5 / 90.5) on the overlapping (smoke) tasks — wide tolerance, 50
   samples is noisy.
4. Prints extrapolated full-run node-hours from recorded wall times.

Exit 0 = Gate 3 green.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval import run_matrix  # noqa: E402

ORDERING_TOLERANCE_PT = 5.0
ANCHOR_TOLERANCE_PT = 10.0
PAPER_4K_ANCHORS = {"base_delta_g64": 96.5, "base_sparse_only": 90.5}


def latest_rows_by_config(results_csv: Path) -> Dict[str, Dict[str, str]]:
    if not results_csv.exists():
        raise SystemExit(f"{results_csv} does not exist — run the smoke matrix first.")
    latest: Dict[str, Dict[str, str]] = {}
    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            latest[row["config_name"]] = row  # file is append-only: last wins
    return latest


def _acc(row: Optional[Dict[str, str]]) -> Optional[float]:
    if row is None or row.get("accuracy") in (None, ""):
        return None
    return float(row["accuracy"])


def extrapolation_table(night1: List[str], latest: Dict[str, Dict[str, str]],
                        doc: Dict[str, Any]) -> str:
    """Rough full-run estimate: startup cost + eval time scaled by sample-cell count.

    Longer contexts are super-linear in cost, so this is a LOWER bound; it
    exists to size the night-1 budget, not to promise it.
    """
    by_name = {c["name"]: c for c in doc["configs"]}
    smoke = doc["smoke_overrides"]
    smoke_cells = len(smoke["context_lengths"]) * len(smoke["tasks"]) * smoke["n_samples"]
    lines = ["config\tstatus\tsmoke_wall_s\tscale\test_full_h"]
    total_h = 0.0
    for name in night1:
        row = latest.get(name)
        full = by_name[name]
        full_cells = len(full["context_lengths"]) * len(full["tasks"]) * full["n_samples"]
        scale = full_cells / smoke_cells
        if row is None or row.get("wall_time_s") in (None, ""):
            lines.append(f"{name}\t{'missing' if row is None else row['status']}\t-\t{scale:.1f}\t-")
            continue
        wall = float(row["wall_time_s"])
        eval_s = 0.0
        try:
            cells = json.loads(row["per_task_json"] or "{}").get("cells", [])
            eval_s = sum(c.get("seconds", 0.0) for c in cells)
        except (ValueError, TypeError):
            pass
        startup_s = max(wall - eval_s, 0.0)
        est_h = (startup_s + eval_s * scale) / 3600.0
        if row["status"] == "ok":
            total_h += est_h
        lines.append(f"{name}\t{row['status']}\t{wall:.0f}\t{scale:.1f}\t{est_h:.2f}")
    lines.append(f"TOTAL estimated full night-1 node-hours (ok configs, lower bound, "
                 f"same node as smoke): {total_h:.2f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=run_matrix.RESULTS_CSV)
    ap.add_argument("--experiments", type=Path, default=run_matrix.EXPERIMENTS_YAML)
    ap.add_argument("--require-all-ok", action="store_true",
                    help="spec-literal Gate 3: unsupported rows also fail the gate")
    args = ap.parse_args(argv)

    doc = run_matrix.load_experiments(args.experiments)
    night1 = doc["groups"]["night1_all"]
    latest = latest_rows_by_config(args.results)

    failures: List[str] = []
    warnings: List[str] = []

    # 1. every night1 config has a row; status ok (unsupported tolerated w/ warning)
    for name in night1:
        row = latest.get(name)
        if row is None:
            failures.append(f"{name}: no results.csv row (it didn't happen)")
        elif row["status"] == "ok":
            pass
        elif row["status"] == "unsupported" and not args.require_all_ok:
            warnings.append(f"{name}: status=unsupported ({row.get('error')}) — "
                            "needs WP-1/WP-3 server support; Gate 3 provisionally "
                            "tolerates this. Re-run with --require-all-ok once it lands.")
        else:
            failures.append(f"{name}: status={row['status']} error={row.get('error')!r}")

    # 2. ordering sanity at 4K (smoke rows are 4K-only)
    dense = _acc(latest.get("base_dense_fa2"))
    delta = _acc(latest.get("base_delta_g64"))
    sparse = _acc(latest.get("base_sparse_only"))
    if None in (dense, delta, sparse):
        failures.append(f"ordering check needs accuracies for dense/delta/sparse, got "
                        f"{dense}/{delta}/{sparse}")
    else:
        if not dense >= delta:
            failures.append(f"ordering: acc(dense)={dense:.2f} < acc(delta)={delta:.2f}")
        if not delta >= sparse - ORDERING_TOLERANCE_PT:
            failures.append(f"ordering: acc(delta)={delta:.2f} < acc(sparse)-"
                            f"{ORDERING_TOLERANCE_PT} = {sparse - ORDERING_TOLERANCE_PT:.2f}")

    # 3. paper 4K anchors, +/-10pt on the overlapping (smoke) tasks
    for name, anchor in PAPER_4K_ANCHORS.items():
        acc = _acc(latest.get(name))
        if acc is None:
            failures.append(f"{name}: no accuracy for anchor check")
        elif abs(acc - anchor) > ANCHOR_TOLERANCE_PT:
            failures.append(f"{name}: accuracy {acc:.2f} outside paper anchor "
                            f"{anchor} +/- {ANCHOR_TOLERANCE_PT}")

    # 4. extrapolated full-run node-hours
    print("[check_smoke] full-run extrapolation from smoke wall times:")
    print(extrapolation_table(night1, latest, doc))

    for w in warnings:
        print(f"[check_smoke] WARN: {w}")
    if failures:
        print("[check_smoke] GATE 3 FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[check_smoke] GATE 3 PASS"
          + (" (with unsupported-config warnings above)" if warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
