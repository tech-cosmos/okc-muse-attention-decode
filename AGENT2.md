# Agent 2 (second Claude agent): batch 16/32 sliding layers

You are the SECOND Claude agent on this problem, added by the human's lead agent on 2026-10-25 04:15 UTC to work
in parallel with agent 1 (which runs in ~/repo on `main`). CLAUDE.md is your brief for goal, rules, precision,
environment, GPU lock and workflow; this file overrides it where they differ:

- **Workspace:** `~/repo2`, branch `agent2` (risky work on `agent2-exp/<name>`). Push only `agent2*` branches.
  Never push `main`, never force-push, never touch ~/repo. Merge `origin/main` into `agent2` at the start of each
  session so you build on agent 1's latest kernel.
- **Your direction (ignore CLAUDE.md's "Current direction", that is agent 1's):** the batch 16/32 sliding shapes:
  decode 16/32 sliding (H200 ~8.7 / 13.3 us) and verify 16/32 sliding (~12.9 / 17.4 us). They sit 3-4x above
  their memory floors (TASK.md: 1.6/3.3/1.9/3.7 us; H100 gather floors decode 16/32 6.9/10.1 us, NOTES s85) and
  are worth ~0.2-0.45 % overall per us (NOTES s82 sensitivity table): ~5 % overall if all four reach ~gather floor.
  Read STATE.md and grep NOTES.md for every sliding dead end first (s70-s75, s84: cluster merges, Gluon sliding,
  csl at batch 32, verify sliding is consumer-chain-bound, splits/stages sweeps). Think structurally: e.g. a native
  wgmma verify-sliding kernel with low fixed cost, a decode 32 sliding kernel that fits one wave, fusing
  _new_tokens + attention + merge. Agent 1 owns batch-1 shapes and full layers: don't work on those.
- **Budget:** at most 8 checks per UTC day (agent 1 keeps the rest of the account's 20; count both agents'
  lines in ~/COORD.md "Judge runs" and leave at least 4 for agent 1). **No scored runs**: when a check shows your
  change winning its shapes, push the branch and add a line under COORD.md "Ready to merge" (branch, commit,
  check id, per-shape H200 times). Agent 1 merges it into main and submits.
- **Notes:** your log is NOTES-agent2.md (newest at the bottom) on your branch; keep a short STATE-agent2.md
  (<= 80 lines). Don't edit NOTES.md / STATE.md (agent 1's). COORD.md: claims/results/judge runs lines prefixed
  `agent2`.
- Stop if ~/STOP or ~/STOP-agent2 exists.
