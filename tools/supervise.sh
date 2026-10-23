#!/bin/bash
# Keeps both agents' loops (tools/night.sh in ~/repo, tools/night_codex.sh in ~/codex-repo) running
# until ~/STOP exists; restarts a loop that died. Logs to ~/logs/supervise.log.
mkdir -p ~/logs
while [ ! -f ~/STOP ]; do
  if ! pgrep -f "repo/tools/night.sh" > /dev/null; then
    echo "$(date -u +%FT%TZ) starting claude loop" >> ~/logs/supervise.log
    setsid nohup ~/repo/tools/night.sh >> ~/logs/night.out 2>&1 < /dev/null &
  fi
  if [ ! -f ~/STOP-codex ] && ! pgrep -f "night_codex.sh" > /dev/null; then
    echo "$(date -u +%FT%TZ) starting codex loop" >> ~/logs/supervise.log
    setsid nohup ~/codex-repo/tools/night_codex.sh >> ~/logs/night-codex.out 2>&1 < /dev/null &
  fi
  sleep 120
done
echo "$(date -u +%FT%TZ) supervisor stopped" >> ~/logs/supervise.log
