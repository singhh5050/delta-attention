#!/usr/bin/env bash
# Unified per-experiment gate chain: setup+Gate1 -> identity tests -> payload.
# Usage: bash eval/run_wp.sh <mode>
# Modes: wp1 | wp3 | wp2 | wp2train | driftprobe | eval32k | falsify | decsweep
#        | wp2pilot-{delta,dense,detach,all} | t2eval | longbench | ppl32k | gap
#        | train32k | distill | enmc
# Writes one line per stage to ~/wp_status; designed for nohup; self-terminates
# + archives on clean completion when SELF_TERMINATE creds are in the env.
set -uo pipefail
WP="${1:?usage: run_wp.sh <mode>}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SMOKE=""; SMOKE_RAW=""; TESTS=""; TRAIN=""; PROBE=""; PILOT=""; ARM=""; T2EVAL=""; LONGBENCH=""; PPL32K=""; GAP=""; TRAIN32K=""; DISTILL=""; ENMC=""
STATUS=~/wp_status
stage() { echo "$(date -u '+%H:%M:%S') $1" >> "$STATUS"; echo; echo "=== WP: $1 ==="; }
: > "$STATUS"

fetch_adapters() {  # usage: fetch_adapters <arm>... — one source of truth
  python - "$@" <<'PYEOF'
import sys, wandb
api = wandb.Api()
for arm in sys.argv[1:]:
    api.artifact(f"singhh5050-stanford-university/delta-attention/wp2_adapter_{arm}:latest").download(root=f"checkpoints/pilot_{arm}")
    print("fetched", arm)
PYEOF
}

case "$WP" in
  wp1) TESTS="tests/test_stride_offline.py tests/test_variable_stride.py"
       SMOKE="t1_adaptive_thr95,t1_fixed_g32,base_delta_g64" ;;
  wp3) TESTS="tests/test_delta_decode.py"
       SMOKE="t3_sparse_decode,t3_delta_dec_g16,t3_delta_dec_g64" ;;
  wp2) TESTS="tests/test_flex_delta.py" ;;
  wp2train) TESTS="tests/test_flex_delta.py"; TRAIN=1 ;;
  driftprobe) TESTS="tests/test_flex_delta.py"; PROBE=1 ;;
  eval32k) TESTS="tests/test_variable_stride.py tests/test_delta_decode.py"
       SMOKE_RAW="poc32k" ;;
  falsify) TESTS="tests/test_delta_decode.py"
       SMOKE_RAW="p32_delta_dec_g1" ;;
  decsweep) TESTS="tests/test_delta_decode.py"
       SMOKE_RAW="decsweep" ;;
  wp2pilot-delta|wp2pilot-dense|wp2pilot-detach)
       ARM="${WP#wp2pilot-}"; TESTS="tests/test_flex_delta.py"; PILOT=1 ;;
  wp2pilot-all)
       ARM="all"; TESTS="tests/test_flex_delta.py"; PILOT=1 ;;
  t2eval) TESTS=""; T2EVAL=1 ;;
  longbench) TESTS="tests/test_longbench_offline.py"; LONGBENCH=1 ;;
  ppl32k) TESTS=""; PPL32K=1 ;;
  gap) TESTS="tests/test_longbench_offline.py tests/test_delta_decode.py"; GAP=1 ;;
  train32k) TESTS="tests/test_flex_delta.py"; TRAIN32K=1 ;;
  distill) TESTS="tests/test_flex_delta.py"; DISTILL=1 ;;
  enmc) TESTS="tests/test_longbench_offline.py"; ENMC=1 ;;
  *) stage "unknown-wp:FAILED"; exit 1 ;;
esac

stage "setup+gate1:running"
bash env/setup.sh || { stage "setup+gate1:FAILED"; exit 1; }
stage "setup+gate1:PASS"

source .venv/bin/activate
source ~/.delta-env 2>/dev/null || true

if [ -n "$TESTS" ]; then
  stage "wp-tests:running"
  python -m pytest $TESTS -v || { stage "wp-tests:FAILED"; exit 1; }
  stage "wp-tests:PASS"
fi

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

if [ -n "$T2EVAL" ]; then
  stage "adapter-fetch:running"
  fetch_adapters delta dense detach || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "posttrain-ppl:running"
  python eval/ppl_eval.py --arms base,delta,dense,detach --chunks 16 \
    || { stage "posttrain-ppl:FAILED"; exit 1; }
  stage "posttrain-ppl:PASS"
  stage "posttrain-ruler:running"
  python eval/run_matrix.py --configs t2eval || { stage "posttrain-ruler:FAILED"; exit 1; }
  stage "posttrain-ruler:PASS"
fi

if [ -n "$PPL32K" ]; then
  stage "adapter-fetch:running"
  fetch_adapters delta dense detach || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "ppl32k:running"
  python eval/ppl_eval.py --arms base,delta,dense,detach --chunks 32 --seq-len 32768 \
    || { stage "ppl32k:FAILED"; exit 1; }
  stage "ppl32k:PASS"
fi

if [ -n "$LONGBENCH" ]; then
  stage "adapter-fetch:running"
  fetch_adapters delta dense || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "longbench-smoke:running"
  python eval/longbench_eval.py --suite v1 --n-samples 5 --arms base_delta \
    --out results/longbench_smoke.csv \
    || { stage "longbench-smoke:FAILED"; exit 1; }
  stage "longbench-smoke:PASS"
  stage "longbench-v1:running"
  python eval/longbench_eval.py --suite v1 --n-samples 50 \
    || { stage "longbench-v1:FAILED"; exit 1; }
  stage "longbench-v1:PASS"
  stage "longbench-v2:running"
  python eval/longbench_eval.py --suite v2 --n-samples 200 \
    || { stage "longbench-v2:FAILED"; exit 1; }
  stage "longbench-v2:PASS"
  # 32K perplexity lives in the dedicated ppl32k mode (32 chunks, 4 arms) —
  # no second, weaker copy of that stage here
fi

if [ -n "$GAP" ]; then
  stage "adapter-fetch:running"
  fetch_adapters delta dense detach || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  # 2x2 completion: same 32 chunks (deterministic selection), dense forward
  stage "ppl32k-dense:running"
  python eval/ppl_eval.py --forward dense --arms base,delta,dense,detach \
    --chunks 32 --seq-len 32768 || { stage "ppl32k-dense:FAILED"; exit 1; }
  stage "ppl32k-dense:PASS"
  stage "longbench-decode:running"
  python eval/longbench_eval.py --suite v1 --n-samples 50 \
    --arms sparse_dec,delta_dec2,delta_dec16 \
    || { stage "longbench-decode:FAILED"; exit 1; }
  stage "longbench-decode:PASS"
fi

if [ -n "$TRAIN32K" ]; then
  # retrain the pilot arms AT the length where the effect lives. 500 steps at
  # 32K = the 8K pilot's token budget (2000x8192), so "longer sequences" is
  # the only lever that moved. --tag _32k keeps wp2_adapter_<arm>:latest (the
  # 8K adapters every existing eval fetches) untouched.
  stage "train32k-smoke:running"
  python -m delta_attention.train.train_delta --steps 20 --seq-len 32768 \
    --probe-every 10 --arm delta --tag _smoke32k --save-dir checkpoints/smoke_32k \
    || { stage "train32k-smoke:FAILED"; exit 1; }
  stage "train32k-smoke:PASS"
  for A in delta dense detach; do
    stage "train32k-$A:running"
    python -m delta_attention.train.train_delta --steps 500 --seq-len 32768 \
      --probe-every 50 --arm "$A" --tag _32k --save-dir "checkpoints/pilot_${A}_32k" \
      || { stage "train32k-$A:FAILED"; exit 1; }
    stage "train32k-$A:PASS"
  done
  # same deterministic 32 chunks as the 8K-adapter runs -> paired across runs
  stage "ppl32k-32ktrained:running"
  python eval/ppl_eval.py --arms base,delta_32k,dense_32k,detach_32k \
    --chunks 32 --seq-len 32768 || { stage "ppl32k-32ktrained:FAILED"; exit 1; }
  stage "ppl32k-32ktrained:PASS"
  stage "ppl32k-32ktrained-dense:running"
  python eval/ppl_eval.py --forward dense --arms base,delta_32k,dense_32k,detach_32k \
    --chunks 32 --seq-len 32768 || { stage "ppl32k-32ktrained-dense:FAILED"; exit 1; }
  stage "ppl32k-32ktrained-dense:PASS"
fi

if [ -n "$DISTILL" ]; then
  # distill objective (KL to the frozen dense teacher), same dials as the CE
  # pilot (2000 steps @8K, identical data/seed) so it is a 4th comparable arm
  stage "distill-smoke:running"
  python -m delta_attention.train.train_delta --steps 20 --seq-len 8192 \
    --probe-every 10 --arm distill --tag _smoke --save-dir checkpoints/smoke_distill \
    || { stage "distill-smoke:FAILED"; exit 1; }
  stage "distill-smoke:PASS"
  stage "distill-train:running"
  python -m delta_attention.train.train_delta --steps 2000 --seq-len 8192 \
    --probe-every 100 --arm distill --save-dir checkpoints/pilot_distill \
    || { stage "distill-train:FAILED"; exit 1; }
  stage "distill-train:PASS"
  stage "adapter-fetch:running"
  fetch_adapters delta dense detach || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "ppl32k-distill:running"
  python eval/ppl_eval.py --arms base,delta,dense,detach,distill \
    --chunks 32 --seq-len 32768 || { stage "ppl32k-distill:FAILED"; exit 1; }
  stage "ppl32k-distill:PASS"
  stage "ppl32k-distill-dense:running"
  python eval/ppl_eval.py --forward dense --arms base,delta,dense,detach,distill \
    --chunks 32 --seq-len 32768 || { stage "ppl32k-distill-dense:FAILED"; exit 1; }
  stage "ppl32k-distill-dense:PASS"
fi

if [ -n "$ENMC" ]; then
  stage "adapter-fetch:running"
  fetch_adapters delta dense || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "enmc-smoke:running"
  python eval/longbench_eval.py --suite enmc --n-samples 3 --arms base_delta \
    --out results/enmc_smoke.csv \
    || { stage "enmc-smoke:FAILED"; exit 1; }
  stage "enmc-smoke:PASS"
  stage "enmc:running"
  python eval/longbench_eval.py --suite enmc --n-samples 229 \
    --arms base_dense,base_delta,ce_delta,dense_delta \
    || { stage "enmc:FAILED"; exit 1; }
  stage "enmc:PASS"
fi

stage "ALL-DONE"
echo; echo "=== results/results.csv (tail) ==="
tail -n 8 results/results.csv 2>/dev/null || true

# Archive box-local outputs, then self-terminate. Failed chains stay up.
if [ -n "${SELF_TERMINATE:-}" ] && [ -n "${LAMBDA_API_KEY:-}" ] && [ -n "${SELF_INSTANCE_ID:-}" ]; then
  python - <<'PYEOF' || echo "[run_wp] WARN: final-state archive failed (continuing to terminate)"
import glob, os, wandb
run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-attention"),
                 name=f"box_archive_{os.environ.get('SELF_INSTANCE_ID','?')[:8]}",
                 job_type="box-archive")
art = wandb.Artifact("box_final_state", type="box-archive")
for p in [os.path.expanduser("~/wp.log"), os.path.expanduser("~/wp_status")] \
        + glob.glob("results/*.csv") + glob.glob("results/server_logs/*.jsonl"):
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
