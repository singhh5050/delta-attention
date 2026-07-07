#!/usr/bin/env bash
# End-to-end PoC chain (WP-0): setup+Gate1 -> drift GPU tests -> Gate2 ->
# smoke matrix (baselines + drift telemetry) -> Gate3 checks.
# Runs unattended under nohup; appends one line per stage to ~/poc_status.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STATUS=~/poc_status
stage() { echo "$(date -u '+%H:%M:%S') $1" >> "$STATUS"; echo; echo "=== POC: $1 ==="; }
: > "$STATUS"

stage "setup+gate1:running"
bash env/setup.sh || { stage "setup+gate1:FAILED"; exit 1; }
stage "setup+gate1:PASS"

# setup.sh ran inside its own venv; activate it for the rest of the chain
source .venv/bin/activate
source ~/.delta-env 2>/dev/null || true

stage "drift-tests:running"
python -m pytest tests/test_drift_logging.py -v || { stage "drift-tests:FAILED"; exit 1; }
stage "drift-tests:PASS"

stage "gate2:running"
python eval/smoke_e2e.py || { stage "gate2:FAILED"; exit 1; }
stage "gate2:PASS"

stage "smoke-matrix:running"
python eval/run_matrix.py --configs night1_all,base_delta_g128 --smoke \
  || { stage "smoke-matrix:FAILED"; exit 1; }
stage "smoke-matrix:PASS"

stage "gate3-checks:running"
python eval/check_smoke.py || { stage "gate3-checks:FAILED"; exit 1; }
stage "ALL-DONE"

echo
echo "=== results/results.csv ==="
cat results/results.csv
