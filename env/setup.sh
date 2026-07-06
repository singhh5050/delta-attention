#!/usr/bin/env bash
# WP-0 §1 — reproducible environment setup for the delta-attention fork.
#
# Idempotent: safe to re-run; every step checks its own state first.
# Choice: plain python venv (not conda) — all pinned deps ship manylinux
# wheels against CUDA 12.x, so no conda-managed CUDA_HOME toolchain is
# needed; the driver + wheel-bundled CUDA runtime suffice. If you need a
# source build of anything, use a conda env with cudatoolkit and re-run
# this script inside it (it will skip venv creation when VIRTUAL_ENV/conda
# is already active).
#
# Requires: NVIDIA GPU + driver, git, python3.10+, HF_TOKEN with gated
# access to meta-llama/Llama-3.1-8B-Instruct.
#
# Ends by running Gate 1 (tests/test_math_identities.py) with per-test
# PASS/FAIL output. Any failure aborts (set -euo pipefail): a setup that
# dies loudly is a success of the system.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
RULER_DIR="${REPO_ROOT}/third_party/RULER"
RULER_LOCK="${REPO_ROOT}/third_party/RULER.lock"
RULER_URL="https://github.com/NVIDIA/RULER.git"
MODEL_STR="meta-llama/Llama-3.1-8B-Instruct"

log() { printf '\n[setup] %s\n' "$*"; }
die() { printf '\n[setup] FATAL: %s\n' "$*" >&2; exit 1; }

cd "${REPO_ROOT}"

# ---------------------------------------------------------------- 0. python
if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
  log "using already-active environment: ${VIRTUAL_ENV:-$CONDA_PREFIX}"
  PY=python
else
  PY_BOOT="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3 || true)"
  [[ -n "${PY_BOOT}" ]] || die "no python3 found"
  "${PY_BOOT}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || die "python 3.10+ required, found $(${PY_BOOT} --version)"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "creating venv at ${VENV_DIR} with ${PY_BOOT}"
    "${PY_BOOT}" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  PY=python
fi
log "python: $(${PY} --version) at $(command -v ${PY})"

# ------------------------------------------------------------- 1. pip deps
log "installing pinned requirements (torch==2.8.0, triton==3.4.0, transformers==4.51.3, hip-attn==1.2.9)"
${PY} -m pip install --upgrade pip >/dev/null
${PY} -m pip install -r "${REPO_ROOT}/requirements.txt"
# Harness extras not in the paper repo's pin list (additive; never upgrades pins):
#   wandb                    — mandatory metric sink (master plan)
#   wonderwords html2text nltk tenacity — RULER synthetic data generation deps
${PY} -m pip install wandb wonderwords html2text nltk tenacity

${PY} - <<'EOF'
import torch, triton, transformers, hip_attn  # noqa: F401
assert torch.__version__.startswith("2.8.0"), f"torch pin violated: {torch.__version__}"
assert triton.__version__.startswith("3.4"), f"triton pin violated: {triton.__version__}"
assert transformers.__version__ == "4.51.3", f"transformers pin violated: {transformers.__version__}"
print(f"[setup] pins OK: torch={torch.__version__} triton={triton.__version__} "
      f"transformers={transformers.__version__}")
assert torch.cuda.is_available(), "torch.cuda.is_available() is False — GPU/driver problem"
print(f"[setup] CUDA OK: {torch.cuda.get_device_name(0)}")
EOF

# ------------------------------------------- 2. hip-attn kernel verification
log "verifying hip-attn block_sparse_attention on random CUDA tensors (mirrors delta_attention/llama.py window path)"
${PY} - <<'EOF'
import torch, triton
from hip_attn.v1_2.attention_extend_bsa import block_sparse_attention
from delta_attention.config import Config, get_hip_config

torch.manual_seed(0)
device, dtype = "cuda", torch.bfloat16
B, S, H, HKV, D = 1, 2048, 32, 8, 128
q = torch.randn(B, S, H, D, device=device, dtype=dtype)
k = torch.randn(B, S, HKV, D, device=device, dtype=dtype)
v = torch.randn(B, S, HKV, D, device=device, dtype=dtype)
position_ids = torch.arange(S, device=device).unsqueeze(0)

cfg = Config()
args = get_hip_config(cfg, 0).clone()
args.position_ids = position_ids
args.sm_scale = D ** -0.5
args.rope_cos = args.rope_sin = None
args.block_size_q = args.block_sparse_block_size_q
args.block_size_k = args.stages[-1].stage_chunk_size
args.second_stage_k = 0
args.sink_token_size = 1024
args.sliding_window_size = cfg.sliding_window
args.sliding_window_indices = None

BDST = triton.cdiv(S, args.block_size_q)
BH = B * H
indices = torch.zeros((BH, BDST, 0), dtype=torch.int64, device=device)
ks = torch.zeros((BH, BDST), dtype=torch.int64, device=device)
out = block_sparse_attention(
    q=(q * args.sm_scale).to(q.dtype), k=k, v=v,
    seq_lens=position_ids + 1,
    indices=indices, ks=ks, ks_count=ks.unsqueeze(-1),
    ks_start_end=torch.zeros((BH, BDST, 2), dtype=torch.int64, device=device),
    args=args, access_counter=None, cache_miss_counter=None,
    model_context_length=131072, extend_context_length=131072 * 10,
    offload_update_cache=False,
)
assert out.shape[:2] == (B, S), f"unexpected output shape {tuple(out.shape)}"
assert torch.isfinite(out.float()).all(), "block_sparse_attention produced non-finite values"
print(f"[setup] hip-attn OK: block_sparse_attention output {tuple(out.shape)} finite")
EOF

# ------------------------------------------------- 3. HF gated model access
log "verifying gated access to ${MODEL_STR}"
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN env var is not set. Export a HuggingFace token with access to ${MODEL_STR} (request access on the model card first)."
${PY} - <<EOF
import os, sys
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError
try:
    info = HfApi(token=os.environ["HF_TOKEN"]).model_info("${MODEL_STR}")
except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as e:
    sys.exit(f"[setup] FATAL: HF_TOKEN cannot access ${MODEL_STR}: {e}\n"
             "Request access at https://huggingface.co/${MODEL_STR} and use a token "
             "of an account that has been granted access.")
print(f"[setup] HF access OK: ${MODEL_STR} (sha {info.sha[:12]})")
EOF

# --------------------------------------------------- 4. RULER clone + pin
mkdir -p "${REPO_ROOT}/third_party"
if [[ ! -d "${RULER_DIR}/.git" ]]; then
  log "cloning NVIDIA RULER into ${RULER_DIR}"
  git clone "${RULER_URL}" "${RULER_DIR}"
fi
PIN="$(grep -v '^#' "${RULER_LOCK}" 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
if [[ -z "${PIN}" || "${PIN}" == "PIN_ME" ]]; then
  PIN="$(git -C "${RULER_DIR}" rev-parse HEAD)"
  log "recording RULER pin ${PIN} in ${RULER_LOCK} (first clone; commit this lockfile)"
  {
    echo "# Pinned commit of https://github.com/NVIDIA/RULER vendored at third_party/RULER."
    echo "# Written by env/setup.sh on first clone; every later setup checks out exactly this."
    echo "${PIN}"
  } > "${RULER_LOCK}"
else
  log "checking out pinned RULER commit ${PIN}"
  git -C "${RULER_DIR}" fetch --quiet origin "${PIN}" 2>/dev/null || true
  git -C "${RULER_DIR}" checkout --quiet "${PIN}" \
    || die "cannot check out RULER pin ${PIN}; inspect ${RULER_LOCK} / ${RULER_DIR}"
fi

# RULER task source data (idempotent; needed by scripts/data/prepare.py):
JSON_DIR="${RULER_DIR}/scripts/data/synthetic/json"
if [[ -d "${JSON_DIR}" ]]; then
  if [[ ! -f "${JSON_DIR}/PaulGrahamEssays.json" && -f "${JSON_DIR}/download_paulgraham_essay.py" ]]; then
    log "downloading RULER niah haystack essays"
    (cd "${JSON_DIR}" && ${PY} download_paulgraham_essay.py) \
      || die "RULER essay download failed — niah data generation will not work"
  fi
  if [[ ! -f "${JSON_DIR}/squad.json" && -f "${JSON_DIR}/download_qa_dataset.sh" ]]; then
    log "downloading RULER QA datasets (squad/hotpotqa)"
    (cd "${JSON_DIR}" && bash download_qa_dataset.sh) \
      || die "RULER QA dataset download failed — qa_1/qa_2 generation will not work"
  fi
else
  die "RULER layout unexpected: ${JSON_DIR} missing. Upstream changed; update env/setup.sh and eval/ruler_client.py together."
fi

${PY} -c "import nltk; nltk.download('punkt', quiet=True)" || true

# ------------------------------------------------------------ 5. Gate 1
log "running Gate 1: pytest tests/test_math_identities.py -v (per-test PASS/FAIL below)"
if ${PY} -m pytest "${REPO_ROOT}/tests/test_math_identities.py" -v; then
  log "Gate 1 PASS — environment is good. Next: Gate 2 (python eval/smoke_e2e.py)."
else
  die "Gate 1 FAILED — do not proceed to Gate 2. Write a failure report per WP0 §3."
fi
