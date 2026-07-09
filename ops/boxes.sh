#!/usr/bin/env bash
# THE one command for the three WP boxes. Per box, idempotently:
#   - initial setup still running -> report and wait
#   - WP chain running            -> show status
#   - WP chain DONE               -> show status
#   - died with a log             -> print failure tail + archive, do NOT relaunch
#   - otherwise                   -> checkout the box's branch, pull, launch
#                                    eval/run_wp.sh (gate chain) under nohup
# Usage:  bash boxes.sh          (all boxes)
#         bash boxes.sh wp3      (one box)
set -euo pipefail
echo "boxes.sh starting ($(date '+%H:%M:%S'))"
{ set +x; } 2>/dev/null  # never trace secrets, even under bash -x
source ~/.delta-keys.env

branch_for() {
  case "$1" in
    wp1) echo wp1-dynamic-stride ;;
    wp2|wp2train|driftprobe|wp2pilot-*) echo wp2-trainable-delta ;;
    wp3) echo wp3-delta-decode ;;
    eval32k|falsify|decsweep) echo wp13-eval32k ;;
    *)   echo wp0-infra ;;
  esac
}

mode_for() {  # argument passed to eval/run_wp.sh on the box
  case "$1" in
    eval32k)  echo eval32k ;;
    wp2train) echo wp2train ;;
    *)        echo "$1" ;;
  esac
}

FILTER="${1:-}"
while read -r NAME IP <&3; do
  [ -n "$FILTER" ] && [ "$NAME" != "$FILTER" ] && continue
  BR=$(branch_for "$NAME")
  MODE=$(mode_for "$NAME")
  echo "################ $NAME ($IP) -> $BR [$MODE] ################"
  SSH=(ssh -i ~/.ssh/lambda_delta -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "ubuntu@$IP")

  BOX_ID=$(awk -v n="$NAME" '$1==n{print $2}' ~/.delta-lambda-boxes 2>/dev/null)
  printf 'export HF_TOKEN=%s\nexport WANDB_API_KEY=%s\nexport WANDB_PROJECT=delta-attention\nexport SELF_TERMINATE=1\nexport LAMBDA_API_KEY=%s\nexport SELF_INSTANCE_ID=%s\n' \
    "$HF_TOKEN" "$WANDB_API_KEY" "$LAMBDA_API_KEY" "$BOX_ID" | "${SSH[@]}" 'umask 077 && cat > ~/.delta-env' \
    || { echo "!! $NAME unreachable"; echo; continue; }

  "${SSH[@]}" bash -s -- "$MODE" "$BR" <<'EOS' || echo "!! $NAME remote step failed"
set -e
WP=$1; BR=$2
# don't touch a box whose initial (wp0-infra) setup is still going
if [ -f ~/setup.pid ] && kill -0 "$(cat ~/setup.pid)" 2>/dev/null; then
  echo ">>> initial setup still running (pid $(cat ~/setup.pid)) — rerun boxes.sh when it finishes"
  tail -n 5 ~/setup.log 2>/dev/null || true
  exit 0
fi
# status endpoint (no ssh needed for monitoring): serves ONLY ~/status/,
# which holds symlinks to wp_status and wp.log — never secrets.
if ! pgrep -f "http[.]server 8899" >/dev/null; then
  mkdir -p ~/status
  ln -sf ~/wp_status ~/status/wp_status
  ln -sf ~/wp.log ~/status/wp.log
  nohup python3 -m http.server 8899 --directory ~/status --bind 0.0.0.0 \
    > /dev/null 2>&1 &
  echo ">>> status endpoint started on :8899"
fi
RUNNING=0; [ -f ~/wp.pid ] && kill -0 "$(cat ~/wp.pid)" 2>/dev/null && RUNNING=1
DONE=0; grep -q "ALL-DONE" ~/wp_status 2>/dev/null && DONE=1
if [ "$RUNNING" = 1 ]; then
  echo ">>> $WP chain running (pid $(cat ~/wp.pid))"
elif [ "$DONE" = 1 ]; then
  echo ">>> $WP chain DONE"
else
  if [ -s ~/wp.log ]; then
    echo "!! previous $WP run died — archiving to ~/wp.log.prev (last 15 lines):"
    tail -n 15 ~/wp.log
    mv -f ~/wp.log ~/wp.log.prev
    cp -f ~/wp_status ~/wp_status.prev 2>/dev/null || true
  fi
  [ -d ~/delta-attention ] || git clone --quiet https://github.com/singhh5050/delta-attention.git ~/delta-attention
  cd ~/delta-attention
  git fetch --quiet origin "$BR"
  git checkout --quiet "$BR"
  git pull --quiet origin "$BR"
  source ~/.delta-env
  rm -f ~/setup.pid ~/setup.log
  nohup bash eval/run_wp.sh "$WP" > ~/wp.log 2>&1 &
  echo $! > ~/wp.pid
  echo ">>> $WP chain launched on $BR@$(git rev-parse --short HEAD) (pid $(cat ~/wp.pid))"
fi
echo "-- wp_status:"; cat ~/wp_status 2>/dev/null || echo "(not started)"
echo "-- log tail:"; tail -n 6 ~/wp.log 2>/dev/null || echo "(no wp.log yet)"
EOS
  echo
done 3< ~/.delta-lambda-box-ips
