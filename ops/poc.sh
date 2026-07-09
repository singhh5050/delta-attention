#!/usr/bin/env bash
# THE one command for the PoC. Idempotent:
#   - chain running        -> show status
#   - chain done (ALL-DONE)-> show status
#   - otherwise            -> pull latest code, wipe stale logs, (re)launch, show status
# Usage:  bash poc.sh
set -euo pipefail
IP=$(cat ~/.delta-lambda-instance-ip)
SSH=(ssh -i ~/.ssh/lambda_delta -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "ubuntu@$IP")

"${SSH[@]}" 'set -e
  RUNNING=0
  if [ -f ~/poc.pid ] && kill -0 "$(cat ~/poc.pid)" 2>/dev/null; then RUNNING=1; fi
  DONE=0
  grep -q "ALL-DONE" ~/poc_status 2>/dev/null && DONE=1

  if [ "$RUNNING" = 0 ] && [ "$DONE" = 0 ]; then
    if [ -s ~/poc.log ]; then
      echo "!!!!! PREVIOUS RUN DIED — its status and last 60 log lines (archived to ~/poc.log.prev):"
      echo "-- previous poc_status:"; cat ~/poc_status 2>/dev/null
      echo "-- previous poc.log tail:"; tail -n 60 ~/poc.log
      echo "!!!!! END OF PREVIOUS FAILURE — NOT relaunching automatically."
      echo "      Paste the above to Claude; relaunch happens on the next poc.sh run after a fix is pushed."
      mv -f ~/poc.log ~/poc.log.prev
      cp -f ~/poc_status ~/poc_status.prev 2>/dev/null || true
      exit 0
    fi
    cd ~/delta-attention && git pull --quiet
    source ~/.delta-env
    rm -f ~/setup.log ~/setup.pid
    echo "$(date -u "+%H:%M:%S") (re)launching chain" > ~/poc_status
    nohup bash eval/run_poc.sh > ~/poc.log 2>&1 &
    echo $! > ~/poc.pid
    echo ">>> chain (re)launched at commit $(git rev-parse --short HEAD) (pid $(cat ~/poc.pid))"
  elif [ "$RUNNING" = 1 ]; then
    echo ">>> chain is running (pid $(cat ~/poc.pid))"
  else
    echo ">>> chain is DONE"
  fi

  echo; echo "== poc_status =="; cat ~/poc_status 2>/dev/null
  echo; echo "== last 25 lines of poc.log =="; tail -n 25 ~/poc.log 2>/dev/null || echo "(log not started yet)"'
