#!/usr/bin/env bash
# Per-WP gate chain: setup+Gate1 -> WP identity tests -> WP smoke configs.
# Usage: bash eval/run_wp.sh <wp1|wp2|wp3>
# Writes one line per stage to ~/wp_status; designed for nohup.
set -uo pipefail
WP="${1:?usage: run_wp.sh <wp1|wp2|wp3>}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STATUS=~/wp_status
stage() { echo "$(date -u '+%H:%M:%S') $1" >> "$STATUS"; echo; echo "=== WP: $1 ==="; }
: > "$STATUS"

case "$WP" in
  wp1) TESTS="tests/test_stride_offline.py tests/test_variable_stride.py"
       SMOKE="t1_adaptive_thr95,t1_fixed_g32,base_delta_g64" ;;
  wp3) TESTS="tests/test_delta_decode.py"
       SMOKE="t3_sparse_decode,t3_delta_dec_g16,t3_delta_dec_g64" ;;
  wp2) TESTS="tests/test_flex_delta.py"
       SMOKE="" ;;  # WP-2's smoke is the T13/T14 gate itself; training comes later
  *) stage "unknown-wp:FAILED"; exit 1 ;;
esac

stage "setup+gate1:running"
bash env/setup.sh || { stage "setup+gate1:FAILED"; exit 1; }
stage "setup+gate1:PASS"

source .venv/bin/activate
source ~/.delta-env 2>/dev/null || true

stage "wp-tests:running"
python -m pytest $TESTS -v || { stage "wp-tests:FAILED"; exit 1; }
stage "wp-tests:PASS"

if [ -n "$SMOKE" ]; then
  stage "wp-smoke:running"
  python eval/run_matrix.py --configs "$SMOKE" --smoke || { stage "wp-smoke:FAILED"; exit 1; }
  stage "wp-smoke:PASS"
fi

stage "ALL-DONE"
echo; echo "=== results/results.csv (tail) ==="
tail -n 8 results/results.csv 2>/dev/null || true
