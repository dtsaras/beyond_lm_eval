#!/usr/bin/env bash
# Poll for the gpt-5.6-sol quota reset, then run the outstanding codex
# re-reviews automatically. Safe to run unattended: it does nothing until the
# quota is actually back, and it re-derives the task list from the verdicts
# each time so it never re-reviews an already dual-signed task.
#
#   nohup bash verification_campaign/auto_resume_codex.sh > /tmp/auto_codex.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
INTERVAL="${INTERVAL:-1800}"          # 30 min between probes

while :; do
  if codex exec --skip-git-repo-check "Reply with exactly: OK" 2>&1 \
       | grep -q "usage limit"; then
    echo "[$(date '+%F %T')] quota still exhausted; sleeping ${INTERVAL}s"
    sleep "$INTERVAL"
    continue
  fi

  echo "[$(date '+%F %T')] quota available — building the task list"
  python3 - <<'PY'
import json, os
C = "verification_campaign"
allt = [t["task"] if isinstance(t, dict) else t
        for t in json.load(open(f"{C}/tasks.json"))]
todo = []
for t in sorted(allt):
    c, o = f"{C}/verdicts/{t}.codex.json", f"{C}/verdicts/{t}.opus5.json"
    cv = json.load(open(c)).get("verdict", "-").upper() if os.path.exists(c) else "-"
    ov = json.load(open(o)).get("verdict", "-").upper() if os.path.exists(o) else "-"
    if not (cv == ov == "GREEN"):
        todo.append(t)
json.dump(todo, open(f"{C}/round7_tasks.json", "w"), indent=1)
print(f"{len(todo)} tasks not yet dual-signed")
PY

  N=$(python3 -c "import json;print(len(json.load(open('verification_campaign/round7_tasks.json'))))")
  if [ "$N" -eq 0 ]; then
    echo "[$(date '+%F %T')] ALL 51 DUAL-SIGNED — nothing to do"
    python3 verification_campaign/refresh.py
    exit 0
  fi

  # Quarantine the stale verdicts for those tasks so the runner regenerates them.
  python3 - <<'PY'
import json, os, shutil, time
C = "verification_campaign"
dst = f"{C}/verdicts_round6_{time.strftime('%Y-%m-%d')}"
os.makedirs(dst, exist_ok=True)
for t in json.load(open(f"{C}/round7_tasks.json")):
    src = f"{C}/verdicts/{t}.codex.json"
    if os.path.exists(src):
        shutil.move(src, f"{dst}/{t}.codex.json")
PY

  echo "[$(date '+%F %T')] running codex over $N tasks"
  CODEX_TASK_LIST=round7_tasks.json CODEX_WORKERS=6 \
    python3 verification_campaign/run_codex_parallel.py
  python3 verification_campaign/refresh.py
  echo "[$(date '+%F %T')] pass complete; re-checking for anything still unsigned"
  sleep 60
done
