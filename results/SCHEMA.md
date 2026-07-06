# results/results.csv schema

One results sink (master plan §5): every completed config run appends **exactly
one row**, keyed by `config_name`. No row = it didn't happen. The file is
append-only; when a config is re-run, the newest row wins (Gate 3 and analysis
read the last row per config).

Granularity decision: the master plan says "one row per config run", so a row
aggregates over the config's full (context_length × task) grid. `accuracy` is
the unweighted mean over all grid cells; the per-cell detail (per context
length and task: accuracy, n, failures, min_pred_chars, seconds) lives in
`per_task_json` and is mirrored to a wandb table.

| column | meaning |
|---|---|
| `config_name` | experiments.yaml row name (the key) |
| `status` | `ok`, `failed` (error recorded, never retried silently), or `unsupported` (config needs WP-1/2/3 server features that have not landed) |
| `mode` | prefill mode: delta / recompute / sparse-only / none (dense FA2) |
| `attn_implementation` | window / hip_attention / flash_attention_2 |
| `gamma` | prefill delta stride (`delta_lambda`) |
| `sliding_window` | StreamingLLM window size |
| `decode_mode` | dense (default) / sparse / delta (WP-3) |
| `gamma_dec` | decode anchor stride (WP-3; empty when N/A) |
| `refresh_policy` | fixed / drift (WP-3; empty when N/A) |
| `stride_policy` | fixed / adaptive / heuristic (WP-1) |
| `context_len` | JSON list of context lengths evaluated |
| `tasks` | JSON list of RULER tasks evaluated |
| `n_samples` | samples per (context, task) cell |
| `accuracy` | mean accuracy (0–100) over all cells |
| `per_task_json` | JSON: `{"cells": [{context_len, task, n, failures, accuracy, min_pred_chars, seconds}...], "unknown_to_server": {...}, "server_notes": [...]}` |
| `samples_per_sec` | completed samples / total eval seconds (excludes server startup) |
| `prefill_ms_p50` | null until prefill timing is separable through the HTTP path |
| `decode_ms_per_token_p50` | null until WP-3 instrumentation lands |
| `oom_count` | occurrences of "CUDA out of memory" across the config's server logs |
| `effective_sparsity` | null until WP-1 lands (then reported for all t1/base rows) |
| `wall_time_s` | end-to-end config wall time incl. server startup |
| `wandb_url` | run URL (empty in WANDB_MODE=offline) |
| `error` | error string for failed/unsupported rows |
| `git_sha` | repo commit the run executed |
| `timestamp` | UTC ISO-8601 append time |
