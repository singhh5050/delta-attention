#!/usr/bin/env bash
# Per-WP gate chain: setup+Gate1 -> WP identity tests -> WP smoke configs.
# Usage: bash eval/run_wp.sh <wp1|wp2|wp3>
# Writes one line per stage to ~/wp_status; designed for nohup.
set -uo pipefail
WP="${1:?usage: run_wp.sh <wp1|wp2|wp3|eval32k>}"
SMOKE=""; SMOKE_RAW=""
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
  eval32k) TESTS="tests/test_variable_stride.py tests/test_delta_decode.py"
       SMOKE_RAW="poc32k" ;;  # 32K rows run as-written (no --smoke shrink)
  falsify) TESTS="tests/test_delta_decode.py"
       SMOKE_RAW="p32_delta_dec_g1" ;;  # gamma_dec=1 must reproduce dense decode
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

if [ -n "$SMOKE_RAW" ]; then
  stage "wp-eval:running"
  python eval/run_matrix.py --configs "$SMOKE_RAW" || { stage "wp-eval:FAILED"; exit 1; }
  stage "wp-eval:PASS"
fi

stage "ALL-DONE"
echo; echo "=== results/results.csv (tail) ==="
tail -n 8 results/results.csv 2>/dev/null || true

# Archive box-local outputs, then self-terminate (ported from wp2 branch).
if [ -n "${SELF_TERMINATE:-}" ] && [ -n "${LAMBDA_API_KEY:-}" ] && [ -n "${SELF_INSTANCE_ID:-}" ]; then
  python - <<'PYEOF' || echo "[run_wp] WARN: final-state archive failed (continuing to terminate)"
import glob, os, wandb
run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                 name=f"box_archive_{os.environ.get('SELF_INSTANCE_ID','?')[:8]}",
                 job_type="box-archive")
art = wandb.Artifact("box_final_state", type="box-archive")
for p in ["results/results.csv", os.path.expanduser("~/wp.log"),
          os.path.expanduser("~/wp_status")] + glob.glob("results/server_logs/*.jsonl"):
    if os.path.exists(p):
        art.add_file(p, name=os.path.basename(p))
run.log_artifact(art)
run.finish()
PYEOF
  echo "[run_wp] self-terminating instance $SELF_INSTANCE_ID in 120s"
  sleep 120
  curl -s -u "$LAMBDA_API_KEY:" \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
    -X POST -H 'Content-Type: application/json' \
    -d "{\"instance_ids\":[\"$SELF_INSTANCE_ID\"]}" || true
fi
