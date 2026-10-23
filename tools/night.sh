#!/bin/bash
# Runs headless Claude Code sessions back to back on this repo until ~/STOP exists.
# Each session follows CLAUDE.md, does one cycle of work, updates NOTES.md, commits and pushes.
cd ~/repo || exit 1
. ~/okc/bin/activate
export OKC_TOKEN="$(cat ~/.okc_token)"
export PATH="$HOME/.local/bin:$HOME/.okc/bin:$PATH"
mkdir -p ~/logs
i=0
while [ ! -f ~/STOP ]; do
  i=$((i + 1))
  start=$(date +%s)
  echo "$(date -u +%FT%TZ) session $i starting" >> ~/logs/night.log
  claude -p --dangerously-skip-permissions --output-format stream-json --verbose \
    "Read CLAUDE.md, then NOTES.md, and do the next cycle of work as CLAUDE.md describes. It is $(date -u +%FT%TZ) (UTC). This is unattended session $i." \
    > ~/logs/session-$i.jsonl 2>&1
  echo "$(date -u +%FT%TZ) session $i exited $? after $(( $(date +%s) - start ))s" >> ~/logs/night.log
  git push -q origin HEAD 2>> ~/logs/night.log
  # A session that ends at once is usually auth, usage limits or an outage: back off.
  if [ $(( $(date +%s) - start )) -lt 300 ]; then sleep 900; else sleep 30; fi
done
echo "$(date -u +%FT%TZ) stopped" >> ~/logs/night.log
