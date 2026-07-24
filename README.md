# Delta Attention Fast and Accurate Sparse Attention Inference by Delta Correction

![Delta Attention Results](./figures/barchart-subset.png)

### Introduction

We found that sparse attention has a problem which hurts performance. The key-sparse attention causes a distirbutional
shift in the attention outputs. As the queries of layer `i+1` depend on the attention outputs of layer `i`, this means that 
even if one were to use a sparse attention prefill and a dense attention decode, the decode may fail to match the proper keys
for a given query due to the distributional shift. 

![Delta Attention Distributional Shift Motivation](./figures/spearman-vs-output-layer-motivation.png)

Delta Attention solves a problem by performing query-sparse (and key-dense) attention for a small subset of query tokens in addition to
the query-dense (and key-sparse) sparse attention method. We then take the difference between the query sparse output and the sparse attention 
output. The difference is then repeated for all missing queries and summed together with the key-sparse attention. The
result is an attention output that is closer in cosine similarity to the full quadratic attention with minimal added
overhead

![Delta Attention Results](./figures/ruler-vs-latency.png)

For more details, please have a look at our [paper here](https://arxiv.org/pdf/2505.11254)

# Usage

We provide a simple implementation with an openai server here. To run the server, execute the following commands

```bash
pip install -r requirements.txt
chmod +x ./run-server-hf.sh
./run-server-hf.sh
```

`run-server-hf.sh` calls `server_hf.py` which starts a simple openai style server at the port specified in
`run-server-hf.sh`. The arguments for `server_hf.py` can be changed according to the following.

```text
usage: server_hf.py [-h] [--model-str MODEL_STR] [--attn-implementation ATTN_IMPLEMENTATION] [--mode MODE] [--hip-attn-args HIP_ATTN_ARGS] [--port PORT] [--host HOST]
                    [--no-trust-remote-code] [--delta-lambda DELTA_LAMBDA] [--sliding-window SLIDING_WINDOW]

options:
  -h, --help            show this help message and exit
  --model-str MODEL_STR
  --attn-implementation ATTN_IMPLEMENTATION
  --mode MODE
  --hip-attn-args HIP_ATTN_ARGS
  --port PORT
  --host HOST
  --no-trust-remote-code
  --delta-lambda DELTA_LAMBDA
  --sliding-window SLIDING_WINDOW
```

# Citation

```
``@inproceedings{willette2025delta,
  title     = {Delta Attention: Fast and Accurate Sparse Attention Inference by Delta Correction},
  author    = {Willette, Jeffrey and Lee, Heejun and Hwang, Sung Ju},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  month     = {December},
  eprint    = {2505.11254},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2505.11254}
}`

## Extension harness (WP-0)

Infrastructure for the extension work packages (see `docs/00_MASTER_PLAN.md`;
the experiment matrix is `experiments.yaml` — its values are human-edited only).

### Setup (fresh GPU instance)

```bash
export HF_TOKEN=...          # gated access to meta-llama/Llama-3.1-8B-Instruct
export WANDB_ENTITY=... WANDB_PROJECT=...
bash env/setup.sh            # installs pins, verifies hip-attn + HF access,
                             # clones RULER at third_party/RULER.lock's pin,
                             # ends by running Gate 1 (per-test PASS/FAIL)
```

### Running experiments

```bash
# Gate 2 — single-sample end-to-end through the real server + RULER client:
python -m pytest tests/  # offline gates
# full reproduction guide for the fork's results: REPRODUCING.md

# Full runs (only after Gates 1–3 are green):
python eval/run_matrix.py --configs night1_all
python eval/run_matrix.py --configs t3_full,t1_full
```

`--configs` accepts config names, group names (from `experiments.yaml`
`groups:`), or `all`. `--smoke` applies `smoke_overrides` verbatim — same code
path, smaller numbers; there are no separate smoke scripts.

### Adding a config

Add a row under `configs:` in `experiments.yaml` (copy an existing row; use
`<<: *defaults`). The runner passes YAML keys that exist as
`delta_attention/config.py::Config` fields to the server as CLI flags; other
keys are recorded in results and wandb. A config that needs a server feature
that hasn't landed yet (e.g. `decode_mode: delta` before WP-3) is recorded as
`status=unsupported` and skipped — it becomes runnable automatically once the
Config field lands.

### Where results land

- `results/results.csv` — exactly one row per config run (schema:
  `results/SCHEMA.md`); server logs under `results/server_logs/`.
- wandb — one run per config (entity/project from `WANDB_ENTITY`/
  `WANDB_PROJECT`), logging the mandatory metric set from the master plan.

Every entrypoint runs the startup validation gate
(`delta_attention/validation.py`) in its first minute and exits(1) on failure.
