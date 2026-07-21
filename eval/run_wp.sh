#!/usr/bin/env bash
# Unified per-experiment gate chain: setup+Gate1 -> identity tests -> payload.
# Usage: bash eval/run_wp.sh <mode>
# Modes: wp1 | wp3 | wp2 | wp2train | driftprobe | eval32k | falsify | decsweep
#        | wp2pilot-{delta,dense,detach,all} | t2eval | longbench | ppl32k | gap
#        | train32k | distill | distill2 | enmc | specdec | specdec3 (2xGPU)
# Writes one line per stage to ~/wp_status; designed for nohup; self-terminates
# + archives on clean completion when SELF_TERMINATE creds are in the env.
set -uo pipefail
WP="${1:?usage: run_wp.sh <mode>}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SMOKE=""; SMOKE_RAW=""; TESTS=""; TRAIN=""; PROBE=""; PILOT=""; ARM=""; T2EVAL=""; LONGBENCH=""; PPL32K=""; GAP=""; TRAIN32K=""; DISTILL=""; ENMC=""; DISTILL2=""; SPECDEC=""; DISTILL3=""; BENCH32K=""; MMLU=""; GRADSCALE=""; SEEDS32K=""; SEEDSDISTILL=""; SPECDEC2=""; SPECDEC3=""; SDTIMING=""; TRIAD=""; TRAINBENCH=""; MODEL2=""
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
q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu",
                    "--format=csv,noheader,nounits"],
                   capture_output=True, text=True).stdout.strip().split("\n")[0]
sm, sm_max, temp = [int(x) for x in q.split(",")[:3]]
print(f"[preflight] under load: {sm}/{sm_max} MHz, {temp}C", flush=True)
assert sm >= 0.7 * sm_max, f"GPU THROTTLED: {sm}/{sm_max} MHz — bad box, relaunch elsewhere"
assert temp <= 82, f"GPU HOT: {temp}C under 30s load — cooling problem, relaunch"
PYEOF
  stage "$ST:PASS"
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
  distill2) TESTS="tests/test_flex_delta.py"; DISTILL2=1 ;;
  distill3) TESTS="tests/test_flex_delta.py"; DISTILL3=1 ;;
  # benchmarks for the 32K-trained adapters + distill_dft (per-sample logged),
  # THEN the dft+CE training cell — bench first so a train failure can't
  # cost the evals
  bench32k) TESTS="tests/test_longbench_offline.py tests/test_flex_delta.py"
            BENCH32K=1; DISTILL3=1 ;;
  mmlu) TESTS="tests/test_longbench_offline.py"; MMLU=1 ;;
  gradscale) TESTS="tests/test_flex_delta.py"; GRADSCALE=1 ;;
  seeds32k) TESTS="tests/test_flex_delta.py"; SEEDS32K=1 ;;
  seedsdistill) TESTS="tests/test_flex_delta.py"; SEEDSDISTILL=1 ;;
  enmc) TESTS="tests/test_longbench_offline.py"; ENMC=1 ;;
  specdec) TESTS="tests/test_longbench_offline.py tests/test_delta_decode.py"; SPECDEC=1 ;;
  specdec2) TESTS="tests/test_specdec_offline.py tests/test_delta_decode.py"; SPECDEC2=1 ;;
  specdec3) TESTS="tests/test_specdec_offline.py tests/test_delta_decode.py"; SPECDEC3=1 ;;
  sdtiming) TESTS="tests/test_specdec_offline.py tests/test_delta_decode.py"; SDTIMING=1 ;;
  triad) TESTS="tests/test_flex_delta.py tests/test_longbench_offline.py"; TRIAD=1 ;;
  trainbench) TESTS="tests/test_flex_delta.py"; TRAINBENCH=1 ;;
  model2) TESTS="tests/test_flex_delta.py"; MODEL2=1 ;;
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
  # smoke ALL THREE arms first (a dense/detach-specific failure must not
  # surface only after the delta arm's hours-long payload run)
  for A in delta dense detach; do
    stage "train32k-smoke-$A:running"
    python -m delta_attention.train.train_delta --steps 20 --seq-len 32768 \
      --probe-every 10 --arm "$A" --tag _smoke32k --no-artifact \
      --save-dir "checkpoints/smoke_32k_$A" \
      || { stage "train32k-smoke-$A:FAILED"; exit 1; }
    stage "train32k-smoke-$A:PASS"
  done
  for A in delta dense detach; do
    stage "train32k-$A:running"
    python -m delta_attention.train.train_delta --steps 500 --seq-len 32768 \
      --probe-every 50 --arm "$A" --tag _32k --save-dir "checkpoints/pilot_${A}_32k" \
      || { stage "train32k-$A:FAILED"; exit 1; }
    stage "train32k-$A:PASS"
  done
  # same deterministic 32 chunks as the 8K-adapter runs -> paired across runs
  run_ppl_2x2 ppl32k-32ktrained base,delta_32k,dense_32k,detach_32k
fi

if [ -n "$DISTILL" ]; then
  # distill objective (KL to the frozen dense teacher), same dials as the CE
  # pilot (2000 steps @8K, identical data/seed) so it is a 4th comparable arm
  stage "distill-smoke:running"
  python -m delta_attention.train.train_delta --steps 20 --seq-len 8192 \
    --probe-every 10 --arm distill --tag _smoke --no-artifact \
    --save-dir checkpoints/smoke_distill \
    || { stage "distill-smoke:FAILED"; exit 1; }
  stage "distill-smoke:PASS"
  # fetch BEFORE the 3h train stage: a bad artifact must fail here, not after
  stage "adapter-fetch:running"
  fetch_adapters delta dense detach || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "distill-train:running"
  python -m delta_attention.train.train_delta --steps 2000 --seq-len 8192 \
    --probe-every 100 --arm distill --save-dir checkpoints/pilot_distill \
    || { stage "distill-train:FAILED"; exit 1; }
  stage "distill-train:PASS"
  run_ppl_2x2 ppl32k-distill base,delta,dense,detach,distill
fi

if [ -n "$DISTILL2" ]; then
  # distill follow-up, two arms: (a) mix = KL to the base teacher + CE
  # (alpha=1), testing whether one objective captures both PG19 adaptation
  # and the tax cut; (b) dft = pure KL to the DENSE-FINETUNED teacher —
  # distill the pipeline onto our best dense-adapted model. Same dials as
  # the pilots otherwise. Fetch first: pilot_dense is the dft teacher.
  stage "adapter-fetch:running"
  fetch_adapters delta dense detach distill || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "distill2-smoke:running"
  python -m delta_attention.train.train_delta --steps 20 --seq-len 8192 \
    --probe-every 10 --arm distill --teacher-checkpoint checkpoints/pilot_dense \
    --tag _smokedft --no-artifact --save-dir checkpoints/smoke_distill_dft \
    || { stage "distill2-smoke:FAILED"; exit 1; }
  stage "distill2-smoke:PASS"
  # the mix config (alpha>0: differentiable chunked CE stacked on the KL
  # graph) is its own code path — smoke it too, not just dft
  stage "distill2-smoke-mix:running"
  python -m delta_attention.train.train_delta --steps 20 --seq-len 8192 \
    --probe-every 10 --arm distill --distill-alpha 1.0 \
    --tag _smokemix --no-artifact --save-dir checkpoints/smoke_distill_mix \
    || { stage "distill2-smoke-mix:FAILED"; exit 1; }
  stage "distill2-smoke-mix:PASS"
  stage "distill2-mix:running"
  python -m delta_attention.train.train_delta --steps 2000 --seq-len 8192 \
    --probe-every 100 --arm distill --distill-alpha 1.0 --tag _mix \
    --save-dir checkpoints/pilot_distill_mix \
    || { stage "distill2-mix:FAILED"; exit 1; }
  stage "distill2-mix:PASS"
  stage "distill2-dft:running"
  python -m delta_attention.train.train_delta --steps 2000 --seq-len 8192 \
    --probe-every 100 --arm distill --teacher-checkpoint checkpoints/pilot_dense \
    --tag _dft --save-dir checkpoints/pilot_distill_dft \
    || { stage "distill2-dft:FAILED"; exit 1; }
  stage "distill2-dft:PASS"
  run_ppl_2x2 ppl32k-distill2 base,delta,dense,detach,distill,distill_mix,distill_dft
fi

BENCH_ARMS=base_dense,base_delta,ce_delta,dense_delta,ce32k_delta,dense32k_delta,detach32k_delta,distill_dft_delta

if [ -n "$BENCH32K" ]; then
  # downstream benchmarks for the 32K-trained adapters (perplexity alone
  # cannot carry the claim): 8 arms x LongBench QA + En.MC, per-sample
  # logged so every between-arm comparison is paired. The 8K arms are
  # re-run to get their per-sample logs (aggregates exist from 07-13/14).
  stage "adapter-fetch-bench:running"
  fetch_adapters delta dense delta_32k dense_32k detach_32k distill_dft \
    || { stage "adapter-fetch-bench:FAILED"; exit 1; }
  stage "adapter-fetch-bench:PASS"
  stage "bench32k-smoke:running"
  python eval/longbench_eval.py --suite v1 --n-samples 3 \
    --arms ce32k_delta,distill_dft_delta --out results/bench32k_smoke.csv \
    || { stage "bench32k-smoke:FAILED"; exit 1; }
  stage "bench32k-smoke:PASS"
  stage "bench32k-v1:running"
  python eval/longbench_eval.py --suite v1 --n-samples 50 --arms "$BENCH_ARMS" \
    || { stage "bench32k-v1:FAILED"; exit 1; }
  stage "bench32k-v1:PASS"
  stage "bench32k-enmc:running"
  python eval/longbench_eval.py --suite enmc --n-samples 229 --arms "$BENCH_ARMS" \
    || { stage "bench32k-enmc:FAILED"; exit 1; }
  stage "bench32k-enmc:PASS"
fi

if [ -n "$DISTILL3" ]; then
  # the last untested cell of (teacher: base|dense-ft) x (loss: KL|KL+CE):
  # KL to the dense-finetuned teacher PLUS CE. The adapted anchor avoids the
  # mix arm's pathology; if this also lands at ~10.37 pipeline ppl, that is
  # evidence for a 2000-step-LoRA capacity ceiling rather than an objective
  # limitation. Same pilot dials as every other arm.
  stage "adapter-fetch:running"
  fetch_adapters delta dense detach distill distill_mix distill_dft \
    || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "distill3-smoke:running"
  python -m delta_attention.train.train_delta --steps 20 --seq-len 8192 \
    --probe-every 10 --arm distill --teacher-checkpoint checkpoints/pilot_dense \
    --distill-alpha 1.0 --tag _smokedftmix --no-artifact \
    --save-dir checkpoints/smoke_distill_dftmix \
    || { stage "distill3-smoke:FAILED"; exit 1; }
  stage "distill3-smoke:PASS"
  stage "distill3-train:running"
  python -m delta_attention.train.train_delta --steps 2000 --seq-len 8192 \
    --probe-every 100 --arm distill --teacher-checkpoint checkpoints/pilot_dense \
    --distill-alpha 1.0 --tag _dftmix --save-dir checkpoints/pilot_distill_dftmix \
    || { stage "distill3-train:FAILED"; exit 1; }
  stage "distill3-train:PASS"
  run_ppl_2x2 ppl32k-distill3 \
    base,delta,dense,detach,distill,distill_mix,distill_dft,distill_dftmix
fi

if [ -n "$ENMC" ]; then
  stage "adapter-fetch:running"
  fetch_adapters delta dense || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "enmc-smoke:running"
  # all four arms at n=3: exercises adapter loading in minutes, not as arms
  # 3-4 of the 229-sample payload hours in
  python eval/longbench_eval.py --suite enmc --n-samples 3 \
    --arms base_dense,base_delta,ce_delta,dense_delta \
    --out results/enmc_smoke.csv \
    || { stage "enmc-smoke:FAILED"; exit 1; }
  stage "enmc-smoke:PASS"
  stage "enmc:running"
  python eval/longbench_eval.py --suite enmc --n-samples 229 \
    --arms base_dense,base_delta,ce_delta,dense_delta \
    || { stage "enmc:FAILED"; exit 1; }
  stage "enmc:PASS"
fi

if [ -n "$SPECDEC" ]; then
  # Jeff's speculative-decoding probe (base model only, no adapters):
  # (a) fill the gamma_dec gap on the QA tasks (2 and 16 already measured),
  # (b) GovReport summarization — long natural-language generation, the
  #     regime where a local-context draft should hold up
  stage "rouge-dep:running"
  # also pinned in requirements.txt; this covers boxes whose venv predates it
  pip install --quiet rouge==1.0.1 || { stage "rouge-dep:FAILED"; exit 1; }
  stage "rouge-dep:PASS"
  stage "specdec-smoke:running"
  python eval/longbench_eval.py --suite govreport --n-samples 2 \
    --arms base_delta,delta_dec8 --out results/govreport_smoke.csv \
    || { stage "specdec-smoke:FAILED"; exit 1; }
  stage "specdec-smoke:PASS"
  stage "qa-gamma-sweep:running"
  python eval/longbench_eval.py --suite v1 --n-samples 50 \
    --arms delta_dec4,delta_dec8 \
    || { stage "qa-gamma-sweep:FAILED"; exit 1; }
  stage "qa-gamma-sweep:PASS"
  stage "govreport:running"
  python eval/longbench_eval.py --suite govreport --n-samples 50 \
    --arms base_dense,base_delta,sparse_dec,delta_dec2,delta_dec4,delta_dec8,delta_dec16 \
    || { stage "govreport:FAILED"; exit 1; }
  stage "govreport:PASS"
fi

if [ -n "$SPECDEC2" ]; then
  # TRUE speculative decoding: draft (sparse/delta) + dense verify, exact
  # greedy parity enforced. Smoke MUST pass --require-exact before the grid.
  stage "adapter-fetch:running"
  fetch_adapters delta_32k distill_dft || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "specdec2-smoke:running"
  python eval/specdec_eval.py --suite govreport --n-samples 5 \
    --drafts sparse,delta --blocks 4 --weights base \
    --exact-check-n 5 --min-parity-prefix 24 --out results/specdec_smoke.csv \
    || { stage "specdec2-smoke:FAILED"; exit 1; }
  stage "specdec2-smoke:PASS"
  stage "specdec2-base-grid:running"
  python eval/specdec_eval.py --suite govreport --n-samples 20 \
    --drafts sparse,delta --blocks 2,4,8 --weights base \
    --min-parity-prefix 24 \
    || { stage "specdec2-base-grid:FAILED"; exit 1; }
  stage "specdec2-base-grid:PASS"
  stage "specdec2-trained:running"
  python eval/specdec_eval.py --suite govreport --n-samples 20 \
    --drafts sparse,delta --blocks 4 --weights ce32k,dft \
    --min-parity-prefix 24 \
    || { stage "specdec2-trained:FAILED"; exit 1; }
  stage "specdec2-trained:PASS"
  stage "specdec2-qa:running"
  python eval/specdec_eval.py --suite qa --n-samples 40 \
    --drafts sparse,delta --blocks 4 --weights base \
    --min-parity-prefix 24 \
    || { stage "specdec2-qa:FAILED"; exit 1; }
  stage "specdec2-qa:PASS"
fi

if [ -n "$SPECDEC3" ]; then
  # Definitive spec-decode grid on a 2x H100 box, one shard per GPU running
  # concurrently: GPU0 = QA full grid; GPU1 = GovReport grid + RULER negative
  # control. Every stage (smokes included) is parity-gated; the run fails if
  # EITHER shard fails, so a failed box stays up for debugging.
  N_GPU=$(nvidia-smi -L | wc -l)
  [ "$N_GPU" -ge 2 ] || { stage "specdec3-gpucheck:FAILED"; exit 1; }
  stage "adapter-fetch:running"
  fetch_adapters delta_32k distill_dft distill_dftmix \
    || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  W_ALL="base,ce32k,dft,dftmix"
  (
    export CUDA_VISIBLE_DEVICES=0
    stage "sd3-qa-smoke:running"
    python eval/specdec_eval.py --suite qa --n-samples 4 \
      --drafts sparse,delta --blocks 4 --weights base \
      --exact-check-n 4 --min-parity-prefix 24 \
      --out results/sd3_qa_smoke.csv \
      --samples-out results/sd3_qa_smoke_samples.csv \
      || { stage "sd3-qa-smoke:FAILED"; exit 1; }
    stage "sd3-qa-smoke:PASS"
    stage "sd3-qa-grid:running"
    python eval/specdec_eval.py --suite qa --n-samples 40 \
      --drafts sparse,delta --blocks 2,4,8 --weights "$W_ALL" \
      --min-parity-prefix 24 \
      --out results/sd3_qa.csv --samples-out results/sd3_qa_samples.csv \
      || { stage "sd3-qa-grid:FAILED"; exit 1; }
    stage "sd3-qa-grid:PASS"
  ) &
  QA_PID=$!
  (
    export CUDA_VISIBLE_DEVICES=1
    stage "sd3-gov-smoke:running"
    python eval/specdec_eval.py --suite govreport --n-samples 3 \
      --drafts sparse,delta --blocks 4 --weights base \
      --exact-check-n 3 --min-parity-prefix 24 \
      --out results/sd3_gov_smoke.csv \
      --samples-out results/sd3_gov_smoke_samples.csv \
      || { stage "sd3-gov-smoke:FAILED"; exit 1; }
    stage "sd3-gov-smoke:PASS"
    stage "sd3-gov-grid:running"
    python eval/specdec_eval.py --suite govreport --n-samples 20 \
      --drafts sparse,delta --blocks 2,4,8 --weights base,ce32k \
      --min-parity-prefix 24 \
      --out results/sd3_gov.csv --samples-out results/sd3_gov_samples.csv \
      || { stage "sd3-gov-grid:FAILED"; exit 1; }
    stage "sd3-gov-grid:PASS"
    # trained drafts at the headline block size only (the training effect is
    # established at K=4; full-K for dft/dftmix would add ~1.5h for low value)
    stage "sd3-gov-trained:running"
    python eval/specdec_eval.py --suite govreport --n-samples 20 \
      --drafts sparse,delta --blocks 4 --weights dft,dftmix \
      --min-parity-prefix 24 \
      --out results/sd3_gov.csv --samples-out results/sd3_gov_samples.csv \
      || { stage "sd3-gov-trained:FAILED"; exit 1; }
    stage "sd3-gov-trained:PASS"
    # negative control: needle retrieval — the draft's local context cannot
    # contain the needle, so drafted-position acceptance should collapse on
    # the answer span (Jeff: RULER is impossible-by-construction for this)
    stage "sd3-ruler:running"
    python eval/specdec_eval.py --suite ruler --n-samples 10 \
      --drafts sparse,delta --blocks 4,8 --weights base \
      --min-parity-prefix 24 \
      --out results/sd3_ruler.csv --samples-out results/sd3_ruler_samples.csv \
      || { stage "sd3-ruler:FAILED"; exit 1; }
    stage "sd3-ruler:PASS"
  ) &
  GOV_PID=$!
  wait "$QA_PID"; QA_RC=$?
  wait "$GOV_PID"; GOV_RC=$?
  if [ "$QA_RC" -ne 0 ] || [ "$GOV_RC" -ne 0 ]; then
    stage "specdec3:FAILED"; exit 1
  fi
  stage "specdec3:PASS"
fi

if [ -n "$MODEL2" ]; then
  # Model-2 replication (07-21, rewritten after review wf_d0eb0869):
  # Qwen3-14B, STRICTLY SEQUENTIAL on ONE GPU — the first draft ran bench
  # cells concurrently with training on the same host, violating the O4
  # idle-box rule. Quality = template-free ppl 2x2. OOM fallback: set
  # M2=Qwen/Qwen3-8B and relaunch (every stage reads $M2).
  M2="${M2_MODEL:-Qwen/Qwen3-14B}"
  export CUDA_VISIBLE_DEVICES=0
  gpu_preflight "m2-gpupreflight"
  # port gate v2: (a) OUR dense forward must match VANILLA transformers to
  # bf16 noise — catches shared-path port bugs (wrong norm placement, rope,
  # weight loading) that a delta-vs-dense comparison cannot see because
  # both arms share _qkv; (b) pipeline tax must be sane.
  stage "m2-portgate:running"
  M2="$M2" python - <<'PYEOF' || { stage "m2-portgate:FAILED"; exit 1; }
import gc, os, torch
M2 = os.environ["M2"]
from datasets import load_dataset
import transformers

tok = transformers.AutoTokenizer.from_pretrained(M2)
ds = load_dataset("emozilla/pg19", split="test", streaming=True)
toks = tok.encode(next(iter(ds))["text"], add_special_tokens=False)[:8192]
ids = torch.tensor([toks], device="cuda")

# reference: vanilla transformers, sdpa
ref = transformers.AutoModelForCausalLM.from_pretrained(
    M2, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
with torch.no_grad():
    l_ref = ref(input_ids=ids, labels=ids, use_cache=False).loss.item()
del ref; gc.collect(); torch.cuda.empty_cache()

from delta_attention.config import Config
from delta_attention.sample import init_model
cfg = Config(); cfg.model_str = M2
cfg.attn_implementation = "window"; cfg.mode = "delta"
cfg.delta_lambda = 64; cfg.sliding_window = 2048
cfg.attn_implementation_original = cfg.attn_implementation
model, _ = init_model(cfg)
model.config.log_drift = False; model.config.detach_delta = False
model.config.use_cache = False
model.eval().cuda()
model.config._attn_implementation = "sdpa"
with torch.no_grad():
    l_dense = model(input_ids=ids, labels=ids, use_cache=False).loss.item()
model.config._attn_implementation = "flex_delta_train"
with torch.no_grad():
    l_delta = model(input_ids=ids, labels=ids, use_cache=False).loss.item()
tax = l_delta - l_dense
print(f"[portgate] vanilla {l_ref:.4f} | ours-dense {l_dense:.4f} "
      f"(diff {abs(l_dense-l_ref):.5f}) | pipeline {l_delta:.4f} "
      f"(tax {tax:.4f})", flush=True)
assert abs(l_dense - l_ref) < 0.01, (
    f"PORT BROKEN: our dense forward diverges from vanilla transformers "
    f"by {abs(l_dense-l_ref):.5f} — norm/rope/loading bug")
assert 0.0 < tax < 0.3, f"PORT SUSPECT: pipeline tax {tax:.4f} outside (0, 0.3)"
PYEOF
  stage "m2-portgate:PASS"
  for SL in 8192 32768; do
    stage "m2-bench-delta-$SL:running"
    python -m delta_attention.train.train_delta --bench --steps 35 \
      --bench-warmup 5 --seq-len "$SL" --arm delta --model "$M2" \
      --probe-every 1000000 --no-artifact --tag "_m2b$SL" \
      --save-dir "checkpoints/m2b_delta_$SL" \
      || { stage "m2-bench-delta-$SL:FAILED"; exit 1; }
    stage "m2-bench-delta-$SL:PASS"
    stage "m2-bench-densefa2-$SL:running"
    python -m delta_attention.train.train_delta --bench --steps 35 \
      --bench-warmup 5 --seq-len "$SL" --arm dense \
      --dense-impl flash_attention_2 --model "$M2" \
      --probe-every 1000000 --no-artifact --tag "_m2b${SL}fa2" \
      --save-dir "checkpoints/m2b_dense_$SL" \
      || { stage "m2-bench-densefa2-$SL:FAILED"; exit 1; }
    stage "m2-bench-densefa2-$SL:PASS"
  done
  stage "m2-train-smoke:running"
  python -m delta_attention.train.train_delta --steps 20 --seq-len 32768 \
    --probe-every 10 --arm delta --model "$M2" --no-artifact \
    --tag _m2smoke --save-dir checkpoints/m2_smoke \
    || { stage "m2-train-smoke:FAILED"; exit 1; }
  stage "m2-train-smoke:PASS"
  for A in delta dense; do
    stage "m2-train-$A:running"
    python -m delta_attention.train.train_delta --steps 500 \
      --seq-len 32768 --probe-every 100 --arm "$A" --model "$M2" \
      --tag "_32k_q3" --save-dir "checkpoints/pilot_${A}_32k_q3" \
      || { stage "m2-train-$A:FAILED"; exit 1; }
    stage "m2-train-$A:PASS"
  done
  stage "m2-ppl:running"
  python eval/ppl_eval.py --arms base,delta_32k_q3,dense_32k_q3 \
    --chunks 32 --seq-len 32768 --model "$M2" \
    || { stage "m2-ppl:FAILED"; exit 1; }
  stage "m2-ppl:PASS"
  stage "m2-ppl-dense:running"
  python eval/ppl_eval.py --forward dense \
    --arms base,delta_32k_q3,dense_32k_q3 \
    --chunks 32 --seq-len 32768 --model "$M2" \
    || { stage "m2-ppl-dense:FAILED"; exit 1; }
  stage "m2-ppl-dense:PASS"
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

if [ -n "$TRIAD" ]; then
  # Paper-core triad on a 2xGPU box (07-17 plan):
  #   GPU0 = T1 training-efficiency benchmark (fwd/bwd/step ms, tok/s, peak
  #          mem; delta/dense/detach x 8K/32K, warm+synced, symmetric loop)
  #          then T2 downstream retention (LongBench v1 paired force-dense +
  #          En.MC for the not-yet-run arms).
  #   GPU1 = T3 second-corpus replication (arXiv: smoke -> delta 32K ->
  #          dense 32K -> paired ppl 2x2 on held-out arxiv chunks).
  # Failure of EITHER shard keeps the box up.
  N_GPU=$(nvidia-smi -L | wc -l)
  [ "$N_GPU" -ge 2 ] || { stage "triad-gpucheck:FAILED"; exit 1; }
  stage "adapter-fetch:running"
  fetch_adapters delta_32k dense_32k distill_dftmix \
    || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  (
    export CUDA_VISIBLE_DEVICES=0
    # T1: bench smoke (3 timed steps must produce a CSV row) then the grid
    stage "t1-smoke:running"
    python -m delta_attention.train.train_delta --bench --steps 8 \
      --bench-warmup 5 --seq-len 8192 --arm delta --probe-every 1000000 \
      --no-artifact --tag _benchsmoke --save-dir checkpoints/bench_smoke \
      || { stage "t1-smoke:FAILED"; exit 1; }
    [ -s results/trainbench.csv ] || { stage "t1-smoke:FAILED"; exit 1; }
    stage "t1-smoke:PASS"
    for A in delta dense detach; do
      for SL in 8192 32768; do
        stage "t1-bench-$A-$SL:running"
        python -m delta_attention.train.train_delta --bench --steps 35 \
          --bench-warmup 5 --seq-len "$SL" --arm "$A" --probe-every 1000000 \
          --no-artifact --tag "_bench$SL" \
          --save-dir "checkpoints/bench_${A}_${SL}" \
          || { stage "t1-bench-$A-$SL:FAILED"; exit 1; }
        stage "t1-bench-$A-$SL:PASS"
      done
    done
    # T2a: LongBench v1 QA, force-dense (capability retention, paired CSVs)
    stage "t2-lb-dense:running"
    python eval/longbench_eval.py --suite v1 --n-samples 50 \
      --arms base_dense,ce32k_delta,dense32k_delta --force-dense \
      --out results/triad_lb_dense.csv \
      || { stage "t2-lb-dense:FAILED"; exit 1; }
    stage "t2-lb-dense:PASS"
    # T2b: En.MC for the arms never run (32K-trained + dftmix), pipeline eval
    stage "t2-enmc-gap:running"
    python eval/longbench_eval.py --suite enmc --n-samples 229 \
      --arms ce32k_delta,dense32k_delta,distill_dftmix_delta \
      --out results/triad_enmc.csv \
      || { stage "t2-enmc-gap:FAILED"; exit 1; }
    stage "t2-enmc-gap:PASS"
  ) &
  T12_PID=$!
  (
    export CUDA_VISIBLE_DEVICES=1
    # T3: arXiv loader smoke (also validates loss wiring on the new corpus)
    stage "t3-smoke:running"
    python -m delta_attention.train.train_delta --steps 30 --seq-len 8192 \
      --data-source arxiv --probe-every 15 --arm delta --no-artifact \
      --tag _arxivsmoke --save-dir checkpoints/smoke_arxiv \
      || { stage "t3-smoke:FAILED"; exit 1; }
    stage "t3-smoke:PASS"
    for A in delta dense; do
      stage "t3-train-$A:running"
      python -m delta_attention.train.train_delta --steps 500 \
        --seq-len 32768 --probe-every 100 --arm "$A" --data-source arxiv \
        --tag "_32k_arxiv" --save-dir "checkpoints/pilot_${A}_32k_arxiv" \
        || { stage "t3-train-$A:FAILED"; exit 1; }
      stage "t3-train-$A:PASS"
    done
    stage "t3-ppl:running"
    python eval/ppl_eval.py --arms base,delta_32k_arxiv,dense_32k_arxiv \
      --chunks 32 --seq-len 32768 --data-source arxiv \
      || { stage "t3-ppl:FAILED"; exit 1; }
    stage "t3-ppl:PASS"
    stage "t3-ppl-dense:running"
    python eval/ppl_eval.py --forward dense \
      --arms base,delta_32k_arxiv,dense_32k_arxiv \
      --chunks 32 --seq-len 32768 --data-source arxiv \
      || { stage "t3-ppl-dense:FAILED"; exit 1; }
    stage "t3-ppl-dense:PASS"
  ) &
  T3_PID=$!
  wait "$T12_PID"; T12_RC=$?
  wait "$T3_PID"; T3_RC=$?
  if [ "$T12_RC" -ne 0 ] || [ "$T3_RC" -ne 0 ]; then
    stage "triad:FAILED"; exit 1
  fi
  stage "triad:PASS"
fi

if [ -n "$SDTIMING" ]; then
  # Definitive TIMING-ONLY rerun after the 07-16 review: --warm-baseline
  # (lean sequential dense loop over every prompt, warm-vs-warm, symmetric
  # harness machinery) replaces the cold generate() refs as the speedup
  # denominator. Acceptance numbers from specdec3 stand; this fixes ONLY
  # the tok_per_s comparison. Single GPU.
  stage "adapter-fetch:running"
  fetch_adapters delta_32k distill_dftmix || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "sdt-smoke:running"
  python eval/specdec_eval.py --suite govreport --n-samples 3 \
    --drafts delta --blocks 4 --weights base --warm-baseline \
    --exact-check-n 3 --min-parity-prefix 24 \
    --out results/sdt_smoke.csv --samples-out results/sdt_smoke_samples.csv \
    || { stage "sdt-smoke:FAILED"; exit 1; }
  stage "sdt-smoke:PASS"
  stage "sdt-delta-grid:running"
  python eval/specdec_eval.py --suite govreport --n-samples 20 \
    --drafts delta --blocks 2,4,8 --weights base,ce32k,dftmix \
    --warm-baseline --min-parity-prefix 24 \
    --out results/sdt_gov.csv --samples-out results/sdt_gov_samples.csv \
    || { stage "sdt-delta-grid:FAILED"; exit 1; }
  stage "sdt-delta-grid:PASS"
  stage "sdt-sparse-contrast:running"
  python eval/specdec_eval.py --suite govreport --n-samples 20 \
    --drafts sparse --blocks 4 --weights base,ce32k,dftmix \
    --warm-baseline --min-parity-prefix 24 \
    --out results/sdt_gov.csv --samples-out results/sdt_gov_samples.csv \
    || { stage "sdt-sparse-contrast:FAILED"; exit 1; }
  stage "sdt-sparse-contrast:PASS"
fi

if [ -n "$MMLU" ]; then
  # capability-retention check: MMLU prompts fit inside sink+window, so
  # delta==dense — this measures finetuning damage, not delta adaptation
  stage "adapter-fetch:running"
  fetch_adapters delta dense detach delta_32k dense_32k detach_32k \
    distill distill_mix distill_dft || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  stage "mmlu-smoke:running"
  python eval/longbench_eval.py --suite mmlu --n-samples 20 --arms base_dense,ce_delta \
    --out results/mmlu_smoke.csv || { stage "mmlu-smoke:FAILED"; exit 1; }
  stage "mmlu-smoke:PASS"
  # all arms force_dense on mmlu (delta==dense at these lengths; base_delta
  # would duplicate base_dense exactly, so it is omitted)
  stage "mmlu:running"
  python eval/longbench_eval.py --suite mmlu --n-samples 1000 \
    --arms base_dense,ce_delta,dense_delta,detach32k_delta,ce32k_delta,dense32k_delta,distill_delta,distill_mix_delta,distill_dft_delta \
    || { stage "mmlu:FAILED"; exit 1; }
  stage "mmlu:PASS"
  # cheap rider (~20 min, adapters already fetched): Jeff's directional-bias
  # question — do anchor-row gradients dominate the DIRECTION of the update?
  stage "gradprobe:running"
  python eval/grad_direction_probe.py --arms delta,delta_32k,detach,dense \
    || { stage "gradprobe:FAILED"; exit 1; }
  stage "gradprobe:PASS"
fi

if [ -n "$GRADSCALE" ]; then
  # interventional test of Jeff's 1/gamma idea: scale ONLY the correction
  # branch's backward. 0.125 = 1/sqrt(64), 0.015625 = 1/64. Same 32K dials
  # as train32k so the arms pair with delta_32k/dense_32k.
  for S in 0.125 0.015625; do
    TAGS=$(echo "$S" | sed 's/0\.125/gsqrt/; s/0\.015625/gsinv/')
    stage "gradscale-smoke-$TAGS:running"
    python -m delta_attention.train.train_delta --steps 20 --seq-len 32768 \
      --probe-every 10 --arm delta --delta-grad-scale "$S" \
      --tag "_smoke$TAGS" --no-artifact --save-dir "checkpoints/smoke_$TAGS" \
      || { stage "gradscale-smoke-$TAGS:FAILED"; exit 1; }
    stage "gradscale-smoke-$TAGS:PASS"
    stage "gradscale-$TAGS:running"
    python -m delta_attention.train.train_delta --steps 500 --seq-len 32768 \
      --probe-every 50 --arm delta --delta-grad-scale "$S" \
      --tag "_32k_$TAGS" --save-dir "checkpoints/pilot_delta_32k_$TAGS" \
      || { stage "gradscale-$TAGS:FAILED"; exit 1; }
    stage "gradscale-$TAGS:PASS"
  done
  stage "adapter-fetch:running"
  fetch_adapters delta_32k dense_32k || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  run_ppl_2x2 ppl32k-gradscale \
    base,delta_32k,dense_32k,delta_32k_gsqrt,delta_32k_gsinv
fi

if [ -n "$SEEDS32K" ]; then
  # training-run variance behind the -0.025 32K gap: seeds 1,2 for the two
  # arms that define it (seed 0 = the existing result)
  for SEED in 1 2; do
    for A in delta dense; do
      stage "seeds32k-${A}-s${SEED}:running"
      python -m delta_attention.train.train_delta --steps 500 --seq-len 32768 \
        --probe-every 100 --arm "$A" --data-seed "$SEED" \
        --tag "_32k_s${SEED}" --save-dir "checkpoints/pilot_${A}_32k_s${SEED}" \
        || { stage "seeds32k-${A}-s${SEED}:FAILED"; exit 1; }
      stage "seeds32k-${A}-s${SEED}:PASS"
    done
  done
  run_ppl_2x2 ppl32k-seeds \
    base,delta_32k_s1,dense_32k_s1,delta_32k_s2,dense_32k_s2
fi

if [ -n "$SEEDSDISTILL" ]; then
  # training-run variance behind the 10.37 delta-CE == distill_dft tie
  stage "adapter-fetch:running"
  fetch_adapters dense || { stage "adapter-fetch:FAILED"; exit 1; }
  stage "adapter-fetch:PASS"
  for SEED in 1 2; do
    stage "seedsdistill-delta-s${SEED}:running"
    python -m delta_attention.train.train_delta --steps 2000 --seq-len 8192 \
      --probe-every 200 --arm delta --data-seed "$SEED" \
      --tag "_s${SEED}" --save-dir "checkpoints/pilot_delta_s${SEED}" \
      || { stage "seedsdistill-delta-s${SEED}:FAILED"; exit 1; }
    stage "seedsdistill-delta-s${SEED}:PASS"
    stage "seedsdistill-dft-s${SEED}:running"
    python -m delta_attention.train.train_delta --steps 2000 --seq-len 8192 \
      --probe-every 200 --arm distill --teacher-checkpoint checkpoints/pilot_dense \
      --data-seed "$SEED" --tag "_dft_s${SEED}" \
      --save-dir "checkpoints/pilot_distill_dft_s${SEED}" \
      || { stage "seedsdistill-dft-s${SEED}:FAILED"; exit 1; }
    stage "seedsdistill-dft-s${SEED}:PASS"
  done
  run_ppl_2x2 ppl32k-seedsdistill \
    base,delta_s1,distill_dft_s1,delta_s2,distill_dft_s2
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
