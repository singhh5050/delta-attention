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
  wp2train) TESTS="tests/test_flex_delta.py"
       SMOKE=""; TRAIN=1 ;;
  driftprobe) TESTS="tests/test_flex_delta.py"
       SMOKE=""; PROBE=1 ;;
  wp2pilot-delta|wp2pilot-dense|wp2pilot-detach)
       ARM="${WP#wp2pilot-}"
       TESTS="tests/test_flex_delta.py"; PILOT=1 ;;
  wp2pilot-all)  # all three arms sequentially on one box
       ARM="all"
       TESTS="tests/test_flex_delta.py"; PILOT=1 ;;
  *) stage "unknown-wp:FAILED"; exit 1 ;;
esac
TRAIN="${TRAIN:-}"; PROBE="${PROBE:-}"; PILOT="${PILOT:-}"

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

if [ -n "$TRAIN" ]; then
  stage "train-smoke:running"
  python -m delta_attention.train.train_delta --steps 50 --seq-len 8192 \
    || { stage "train-smoke:FAILED"; exit 1; }
  stage "train-smoke:PASS"
fi

if [ -n "$PROBE" ]; then
  stage "drift-probe-ruler:running"
  python eval/drift_probe.py --data ruler --ruler-task niah_single_1 \
    --context-lengths 32768,65536,131072 --n-docs 3 \
    --out results/drift_probe/probe_ruler.json \
    || { stage "drift-probe-ruler:FAILED"; exit 1; }
  stage "drift-probe-ruler:PASS"
  stage "drift-probe-pg19-long:running"
  python eval/drift_probe.py --data pg19 --context-lengths 65536,131072 --n-docs 3 \
    --out results/drift_probe/probe_pg19_long.json \
    || { stage "drift-probe-pg19-long:FAILED"; exit 1; }
  stage "drift-probe-pg19-long:PASS"
fi

if [ -n "$PILOT" ]; then
  if [ "$ARM" = "all" ]; then ARMS="delta dense detach"; else ARMS="$ARM"; fi
  for A in $ARMS; do
    stage "pilot-$A:running"
    python -m delta_attention.train.train_delta --steps 2000 --seq-len 8192 \
      --probe-every 100 --arm "$A" --save-dir "checkpoints/pilot_$A" \
      || { stage "pilot-$A:FAILED"; exit 1; }
    stage "pilot-$A:PASS"
  done
fi

stage "ALL-DONE"
echo; echo "=== results/results.csv (tail) ==="
tail -n 8 results/results.csv 2>/dev/null || true

# Self-terminate when the chain completes cleanly (idle finished boxes have
# cost ~$19 so far). Requires SELF_TERMINATE=1 + LAMBDA_API_KEY +
# SELF_INSTANCE_ID in ~/.delta-env (boxes.sh provides them). Failed chains
# stay up so logs remain reachable.
if [ -n "${SELF_TERMINATE:-}" ] && [ -n "${LAMBDA_API_KEY:-}" ] && [ -n "${SELF_INSTANCE_ID:-}" ]; then
  # archive everything box-local that isn't already durable (results.csv,
  # chain log/status, raw telemetry sidecars) before the disk disappears
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
  echo "[run_wp] self-terminating instance $SELF_INSTANCE_ID in 120s (wandb flush grace)"
  sleep 120
  curl -s -u "$LAMBDA_API_KEY:" \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
    -X POST -H 'Content-Type: application/json' \
    -d "{\"instance_ids\":[\"$SELF_INSTANCE_ID\"]}" || true
fi
