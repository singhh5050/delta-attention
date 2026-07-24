# Reproducing the headline results

Three self-contained entry points, in the order you'll likely want them.
General notes:
- We log to wandb by default; set `WANDB_MODE=disabled` to run fully
  offline (results still land in the `--out` CSV).
- All numbers we report are in `docs/stats_training_mtp.md` with run IDs;
  the raw CSVs behind every table are archived per-run.

## 1. Attention-layer latency ladder (the 12–47× numbers)

Weightless single attention layer, random inputs, Llama-8B geometry
(32 Q / 8 KV heads, head_dim 128), sink 1024 + window 2048, γ=64.
Needs only torch 2.8 (FlexAttention) — no model downloads, no HF token.

    python3.11 -m venv .venv && . .venv/bin/activate
    pip install torch==2.8.0 wandb
    WANDB_MODE=disabled python eval/anchor_bench.py --seq-lens 8192,32768
    WANDB_MODE=disabled python eval/anchor_bench.py \
        --seq-lens 131072,262144,524288,1048576 \
        --skip anchor-masked,anchor-efficient,anchor-mathonly

All numbers are PER LAYER (multiply by n_layers for a step). The --skip
trio is required at 1M: those cells' mem-efficient backward is known-fatal
there (their clean OOM rows exist in our archives). Expect RAISED-64BIT
rows ≥524K for monolithic flex cells (torch 2.8 int32 indexing) — the
-hc4 head-chunked cells carry the curve, parity-certified in-process.
fwd+bwd at 1M fits no formulation on one 80GB card (dense included).

## 2. Gemma 4 MTP drafter × delta reads (G1)

Needs its own venv: gemma4/gemma4_assistant exist only on transformers
main. Models: `google/gemma-4-31b-it` + `-assistant` (~62GB bf16 — one
80GB card; HF token needed only if the repo is gated for your account).

    python3.11 -m venv .venv-g4 && . .venv-g4/bin/activate
    pip install torch==2.8.0 "git+https://github.com/huggingface/transformers" \
        datasets accelerate wandb sentencepiece protobuf
    # gates first (zero-tolerance parity + native cross-check):
    WANDB_MODE=disabled python eval/gemma4_g1_eval.py --n 2 \
        --tiers 4096,8192 --max-new 64 --arms full --parity-check \
        --out results/g1_smoke.csv
    # the experiment (arms differ only in the KV view the drafter sees):
    WANDB_MODE=disabled python eval/gemma4_g1_eval.py --n 6 \
        --tiers 4096,8192,16384 --max-new 128 --k 5 \
        --arms full,sparse,delta4,deltacorr5 --out results/g1_tiers.csv

Arms: `full` | `sparse` (sink1024+window2048 read) | `deltaN` (full read
every Nth drafter call) | `deltacorrN` (sparse reads + the actual delta
correction: anchor per round computes full & sparse, Δ applied to
subsequent calls via a forward hook — drafter weights untouched).

Known limits, please read before burning GPU-hours:
- **16K is the single-80GB ceiling for the 31B trunk** (fit ladder in
  docs/stats_training_mtp.md §V2: 16K peaks 75.8GB; 32K/65K OOM; CPU
  cache offload is upstream-broken — nondeterministic — on current
  transformers main). For 32K+: 2×H100 with `device_map="auto"` should
  lift it, but that path is UNTESTED by us — treat first results as
  suspect until the parity gate passes there.
- Acceptance is judged from n≥6 tier tables only; smoke-log acceptance
  at n=2 is noise (spread 1.17–1.48 within a tier).
- Aggregation rule: drop any (tier, idx) that has an OOM@*, SKIPPED-*,
  or PROMPT-UNPAIRED marker row before computing per-arm means.

## 3. Training-efficiency benchmark (the 1.22× @32K step numbers)

Full stack (flash-attn, hip-attn, pinned transformers 4.51.3):
`bash env/setup.sh` on a clean CUDA 12.x box builds `.venv` with all
pins, then:

    . .venv/bin/activate
    WANDB_MODE=disabled python eval/swa_bench.py --seq-len 32768 --steps 30

Protocol requirements that materially change numbers if skipped: idle
GPU (nothing co-located), GPU-health preflight (a thermally-throttled
H100 silently produced 2.4×-slow numbers once), warmup ≥5 per variant,
compare delta-flex against dense-**fa2** (sdpa inflates the ratio ~5pts).
