#!/usr/bin/env bash
# Unified per-experiment gate chain: setup+Gate1 -> identity tests -> payload.
# Usage: bash eval/run_wp.sh <mode>
# Modes: trainbench | swabench | anchorbench | evalgaps | gemma4g1
#        | wp2pilot-{delta,dense,detach,all} | t2eval | longbench | ppl32k | gap
#        | train32k | distill | distill2 | enmc | specdec | specdec3 (2xGPU)
# Writes one line per stage to ~/wp_status; designed for nohup; self-terminates
# + archives on clean completion when SELF_TERMINATE creds are in the env.
set -uo pipefail
WP="${1:?usage: run_wp.sh <mode>}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TESTS=""; TRAINBENCH=""; SWABENCH=""; ANCHORBENCH=""; EVALGAPS=""; GEMMA4G1=""
STATUS=~/wp_status
stage() { echo "$(date -u '+%H:%M:%S') $1" >> "$STATUS"; echo; echo "=== WP: $1 ==="; }
: > "$STATUS"

fetch_adapters() {  # usage: fetch_adapters <arm>... — one source of truth
  python - "$@" <<'PYEOF'
import sys, time, wandb
api = wandb.Api()
def hf_fallback(arm, root):
    """Adapters are mirrored to a private HF repo (2026-07-15, after the
    key-rotation incident made wandb artifacts briefly unreachable)."""
    import os
    from huggingface_hub import snapshot_download
    snapshot_download("singhh5050/delta-attention-adapters", repo_type="model",
                      allow_patterns=[f"{arm}/*"], local_dir="checkpoints/_hf",
                      token=os.environ.get("HF_TOKEN"))
    os.makedirs(root, exist_ok=True)
    src = f"checkpoints/_hf/{arm}"
    for f in os.listdir(src):
        os.replace(os.path.join(src, f), os.path.join(root, f))

for arm in sys.argv[1:]:
    root = f"checkpoints/pilot_{arm}"
    for attempt in range(3):  # wandb, then wandb, then the HF mirror
        try:
            if attempt < 2:
                art = api.artifact(f"singhh5050-stanford-university/delta-attention/wp2_adapter_{arm}:latest")
                art.download(root=root)
            else:
                print(f"fetch {arm}: falling back to HF mirror", flush=True)
                hf_fallback(arm, root)
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"fetch {arm} attempt {attempt + 1} failed ({e}); retrying in 30s", flush=True)
            time.sleep(30)
    # record the RESOLVED version+digest next to the weights: ':latest' is a
    # floating alias, and anything trained/evaled against this adapter (e.g.
    # a distill teacher) needs pinned provenance to be reproducible
    with open(f"{root}/WANDB_ARTIFACT", "w") as f:
        f.write(f"wp2_adapter_{arm}:{art.version} digest={art.digest}\n")
    print(f"fetched {arm} -> {art.version} digest={art.digest}")
PYEOF
}

run_ppl_2x2() {  # usage: run_ppl_2x2 <stage_prefix> <arms> — the shared
  # 32-chunk @32K grid, both forwards; ONE copy of the dials so future
  # changes can't silently diverge between modes
  local P=$1 A=$2
  stage "$P:running"
  python eval/ppl_eval.py --arms "$A" --chunks 32 --seq-len 32768 \
    || { stage "$P:FAILED"; exit 1; }
  stage "$P:PASS"
  stage "$P-dense:running"
  python eval/ppl_eval.py --forward dense --arms "$A" --chunks 32 --seq-len 32768 \
    || { stage "$P-dense:FAILED"; exit 1; }
  stage "$P-dense:PASS"
}

gpu_preflight() {  # usage: gpu_preflight <stage-name> — burn-in then clock/temp
  # assertion (box-31 lesson: a throttled H100 gives 2.4x-slow, ratio-
  # distorted numbers with zero errors)
  local ST=$1
  stage "$ST:running"
  python - <<'PYEOF' || { stage "$ST:FAILED"; exit 1; }
import subprocess, time, torch
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
t0 = time.monotonic()
while time.monotonic() - t0 < 30:
    a = (a @ a).clamp(-1, 1)
torch.cuda.synchronize()
out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu",
                      "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout.strip()
for line in out.splitlines():  # validate EVERY GPU (wf_c5d06bb1: [0]-only
    # silently passed multi-GPU boxes with a sick second GPU)
    sm, sm_max, temp = [int(x) for x in line.split(",")[:3]]
    print(f"[preflight] under load: {sm}/{sm_max} MHz, {temp}C", flush=True)
    assert sm >= 0.7 * sm_max, f"GPU THROTTLED: {sm}/{sm_max} MHz — bad box"
    assert temp <= 82, f"GPU HOT: {temp}C under 30s load — cooling problem"
PYEOF
  stage "$ST:PASS"
}

case "$WP" in
  # live modes only — every retired chain mode (wp*, pilots, distill*,
  # specdec*, triad, model2, mtpa, mimo, gemma4-G0, ...) lives in git
  # history; its results are final in docs/STATS.md
  trainbench) TESTS="tests/test_flex_delta.py"; TRAINBENCH=1 ;;
  swabench) TESTS="tests/test_flex_delta.py"; SWABENCH=1 ;;
  anchorbench) TESTS=""; ANCHORBENCH=1 ;;
  # eval gaps: full 16-task English LongBench on the 32K-trained arms +
  # RULER on the same arms
  evalgaps) TESTS="tests/test_longbench_offline.py"; EVALGAPS=1 ;;
  # G1: delta-read the native Gemma 4 drafter via our own draft-verify
  # loop (the shared_kv_states dict is the intervention point)
  gemma4g1) TESTS=""; GEMMA4G1=1 ;;
  *) stage "unknown-wp:FAILED"; exit 1 ;;
esac

stage "setup+gate1:running"
bash env/setup.sh || { stage "setup+gate1:FAILED"; exit 1; }
stage "setup+gate1:PASS"

source .venv/bin/activate
source ~/.delta-env 2>/dev/null || true
# bench-only knobs must never leak into a chain from ambient shell state
unset DELTA_SPARSE_IMPL DELTA_FA2_WINDOW

if [ -n "$TESTS" ]; then
  stage "wp-tests:running"
  python -m pytest $TESTS -v || { stage "wp-tests:FAILED"; exit 1; }
  stage "wp-tests:PASS"
fi

BENCH_ARMS=base_dense,base_delta,ce_delta,dense_delta,ce32k_delta,dense32k_delta,detach32k_delta,distill_dft_delta

if [ -n "$ANCHORBENCH" ]; then
  # component-level decomposition + Jeff's weightless long-context ladder
  # (131K -> 1M) + MTP head-read cost curves. Pure tensor microbench.
  export CUDA_VISIBLE_DEVICES=0  # wf_c5d06bb1: unpinned = contention risk
  gpu_preflight "ab-gpupreflight"
  stage "anchorbench-short:running"
  python eval/anchor_bench.py --seq-lens 8192,32768 \
    || { stage "anchorbench-short:FAILED"; exit 1; }
  stage "anchorbench-short:PASS"
  stage "anchorbench-long:running"
  python eval/anchor_bench.py --seq-lens 131072,262144,524288,1048576 \
    || { stage "anchorbench-long:FAILED"; exit 1; }
  stage "anchorbench-long:PASS"
fi

if [ -n "$SWABENCH" ]; then
  # Sparse-branch kernel diagnostic (Jeff 07-21): flex vs flex-native-GQA vs
  # FA2 sliding window (no sink, timing-only) vs dense-FA2 reference, looped
  # IN-PROCESS per seq-len (one model load, identical weights). Idle box.
  gpu_preflight "swa-gpupreflight"
  for SL in 8192 32768; do
    stage "swabench-$SL:running"
    python eval/swa_bench.py --seq-len "$SL" --steps 30 \
      || { stage "swabench-$SL:FAILED"; exit 1; }
    stage "swabench-$SL:PASS"
  done
fi

if [ -n "$TRAINBENCH" ]; then
  # CLEAN T1 rerun (07-20 review): IDLE box, strictly sequential — the
  # 07-17 numbers were timed concurrently with a training job on the other
  # GPU of the same host. Adds the fa2-dense baseline (the 07-17 dense arm
  # ran sdpa, a kernel confound) and probe-free peak memory.
  gpu_preflight "tb-gpupreflight"
  stage "tb-smoke:running"
  python -m delta_attention.train.train_delta --bench --steps 8 \
    --bench-warmup 5 --seq-len 8192 --arm delta --probe-every 1000000 \
    --no-artifact --tag _tbsmoke --save-dir checkpoints/bench_smoke \
    || { stage "tb-smoke:FAILED"; exit 1; }
  [ -s results/trainbench.csv ] || { stage "tb-smoke:FAILED"; exit 1; }
  stage "tb-smoke:PASS"
  for SL in 8192 32768; do
    for A in delta detach; do
      stage "tb-$A-$SL:running"
      python -m delta_attention.train.train_delta --bench --steps 35 \
        --bench-warmup 5 --seq-len "$SL" --arm "$A" --probe-every 1000000 \
        --no-artifact --tag "_tb$SL" --save-dir "checkpoints/tb_${A}_${SL}" \
        || { stage "tb-$A-$SL:FAILED"; exit 1; }
      stage "tb-$A-$SL:PASS"
    done
    for IMPL in sdpa flash_attention_2; do
      stage "tb-dense-$IMPL-$SL:running"
      python -m delta_attention.train.train_delta --bench --steps 35 \
        --bench-warmup 5 --seq-len "$SL" --arm dense --dense-impl "$IMPL" \
        --probe-every 1000000 --no-artifact --tag "_tb${SL}_$IMPL" \
        --save-dir "checkpoints/tb_dense_${IMPL}_${SL}" \
        || { stage "tb-dense-$IMPL-$SL:FAILED"; exit 1; }
      stage "tb-dense-$IMPL-$SL:PASS"
    done
  done
fi

if [ -n "$EVALGAPS" ]; then
  export CUDA_VISIBLE_DEVICES=0
  gpu_preflight "eg-gpupreflight"
  stage "adapter-fetch:running"
  fetch_adapters delta_32k dense_32k detach_32k \
    || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  # cheap first: RULER (prefill-only scoring) before the generation-heavy
  # LongBench pass, so a LongBench failure can't cost the RULER rows
  stage "ruler32k:running"
  python eval/run_matrix.py --configs ruler32k \
    || { stage "ruler32k:FAILED"; exit 1; }
  stage "ruler32k:PASS"
  stage "lbfull-smoke:running"
  python eval/longbench_eval.py --suite v1full --n-samples 2 \
    --arms base_delta --out results/lbfull_smoke.csv \
    || { stage "lbfull-smoke:FAILED"; exit 1; }
  stage "lbfull-smoke:PASS"
  stage "lbfull:running"
  python eval/longbench_eval.py --suite v1full --n-samples 50 \
    --arms base_dense,base_delta,ce32k_delta,dense32k_delta,detach32k_delta \
    --out results/lbfull.csv \
    || { stage "lbfull:FAILED"; exit 1; }
  stage "lbfull:PASS"
fi

if [ -n "$GEMMA4G1" ]; then
  export CUDA_VISIBLE_DEVICES=0
  # 16K sits ~1GB over the fragmented-allocator ceiling with both models
  # loaded (box-45 v7: OOM asking 674MB with 467MB free at 78.7GB held;
  # diag4 measured 75.8GB peak target-only). expandable_segments removes
  # segment fragmentation; allocator policy only, no semantic effect.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  gpu_preflight "g1-gpupreflight"
  stage "g1-venv:running"
  G4PY="$PWD/.venv-g4/bin/python"
  python3 -m venv .venv-g4 \
    && "$G4PY" -m pip install -q --upgrade pip \
    && "$G4PY" -m pip install -q torch==2.8.0 \
    && "$G4PY" -m pip install -q \
         "git+https://github.com/huggingface/transformers" \
         datasets accelerate wandb sentencepiece protobuf \
    && "$G4PY" -c "from transformers import Gemma4AssistantForCausalLM; print('gemma4_assistant import OK')" \
    || { stage "g1-venv:FAILED"; exit 1; }
  stage "g1-venv:PASS"
  # v7: SINGLE-SHOT prefill everywhere — the same code path certified at
  # 4K/8K (run 92y2luja). Box-45 diags (2026-07-22): --offload is
  # upstream-broken (nondeterministic trunk at ctx>2K); chunked prefill
  # is uncertifiable (13-sigma sliding-KV outliers vs single-shot);
  # fit ladder (diag4): 16K single-shot FITS (75.8GB peak incl. the
  # hidden-states last-token step); 32K and 65K OOM — on 1xH100 with
  # this upstream, 16K is the ceiling, so the long run is 16K ONLY
  # (32K/65K would only churn OOM rows). Box 44's "16K wall" was
  # allocator litter across 48 arm runs, fixed by the b156083 cleanup.
  # Smoke gates: parity (zero-tolerance vs shape-aligned plain greedy),
  # native cross-check. NOTE on acceptance in the smoke logs: at n=2 /
  # max_new 64 the per-prompt spread is huge (box-46 recert: 1.481/1.200
  # at 4K, 1.172/1.462 at 8K) — there is NO usable numeric anchor at
  # smoke params (review a15f950a: quoting single-prompt values here
  # false-flagged healthy runs). The smoke stage gates on parity + the
  # native cross-check ONLY; acceptance regressions are judged from the
  # n=6 arms tables, never the smoke rows.
  stage "g1-smoke:running"
  "$G4PY" eval/gemma4_g1_eval.py --n 2 --tiers 4096,8192 --max-new 64 \
    --arms full --parity-check --out results/g1_smoke.csv \
    || { stage "g1-smoke:FAILED"; exit 1; }
  stage "g1-smoke:PASS"
  # --parity-check STAYS ON at the long tiers: these are the tiers the
  # results come from, so the gate must run where they run (review
  # 2026-07-22: an ungated 16K+ arms run would certify nothing)
  stage "g1-arms:running"
  "$G4PY" eval/gemma4_g1_eval.py --n 6 --tiers 16384 \
    --max-new 128 --k 5 --arms full,sparse,delta2,delta4 \
    --parity-check --out results/g1_tiers.csv \
    || { stage "g1-arms:FAILED"; exit 1; }
  stage "g1-arms:PASS"
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
