# Ops scripts (run from a local machine, not the GPU boxes)

- `boxes.sh` — one idempotent command per named box: places secrets from ~/.delta-keys.env,
  clones/pulls the box's mapped branch, starts the :8899 status endpoint, launches
  eval/run_wp.sh under nohup with self-terminate credentials, or reports status.
- `poc.sh` — single-box variant used for the original WP-0 proof-of-concept chain.

Requires: ~/.delta-keys.env (LAMBDA_API_KEY, HF_TOKEN, WANDB_API_KEY), an SSH key
registered with Lambda, and box name->instance-id/IP files (~/.delta-lambda-boxes,
~/.delta-lambda-box-ips) written at launch time.
