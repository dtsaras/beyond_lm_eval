#!/usr/bin/env bash
# usage: run_codex.sh <BATCH_ID> <task> [task...]
set -u
BID="$1"; shift
CAMP=~/projects/beyond_lm_eval/verification_campaign
mkdir -p "$CAMP/logs"
TASKS="$*"
PROMPT="$(cat "$CAMP/BRIEF.md")

---

Your reviewer id is: codex
Your assigned tasks for batch $BID are: $TASKS

Work through them one at a time. Write each verdict JSON to
verification_campaign/verdicts/<task>.codex.json as soon as you finish that
task, so partial progress survives. Do not modify any other file."
cd ~/projects/beyond_lm_eval
codex exec --skip-git-repo-check "$PROMPT" > "$CAMP/logs/$BID.codex.log" 2>&1
echo "codex $BID done rc=$?"
