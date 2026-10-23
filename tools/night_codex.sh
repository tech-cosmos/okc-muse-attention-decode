#!/bin/bash
# Runs headless Codex sessions back to back on the `codex` branch until ~/STOP or ~/STOP-codex exists.
# Each session follows CODEX.md, does one cycle of work, updates NOTES-codex.md, commits and pushes.
# The model comes from ~/.codex_model if present, else Codex's own default.
cd ~/codex-repo || exit 1
. ~/okc/bin/activate
export OKC_TOKEN="$(cat ~/.okc_token)"
export PATH="$HOME/.local/bin:$HOME/.okc/bin:$PATH"
model=()
[ -s ~/.codex_model ] && model=(-m "$(cat ~/.codex_model)")
mkdir -p ~/logs
i=0
while [ ! -f ~/STOP ] && [ ! -f ~/STOP-codex ]; do
  i=$((i + 1))
  start=$(date +%s)
  echo "$(date -u +%FT%TZ) codex session $i starting" >> ~/logs/night-codex.log
  codex exec --dangerously-bypass-approvals-and-sandbox --json "${model[@]}" \
    "Read CODEX.md, then CLAUDE.md, NOTES.md and NOTES-codex.md, and do the next cycle of work as CODEX.md describes. It is $(date -u +%FT%TZ) (UTC). This is unattended Codex session $i." \
    > ~/logs/codex-$i.jsonl 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) codex session $i exited $? after $(( $(date +%s) - start ))s" >> ~/logs/night-codex.log
  git push -q origin codex 2>> ~/logs/night-codex.log
  # A session that ends at once is usually auth, usage limits or an outage: back off.
  if [ $(( $(date +%s) - start )) -lt 300 ]; then sleep 900; else sleep 30; fi
done
echo "$(date -u +%FT%TZ) codex stopped" >> ~/logs/night-codex.log
