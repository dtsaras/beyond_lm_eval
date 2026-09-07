#!/usr/bin/env bash
set -u
BID="$1"; shift
CAMP=~/projects/beyond_lm_eval/verification_campaign
mkdir -p "$CAMP/logs"
PROMPT="$(cat "$CAMP/BRIEF.md")

---

Your reviewer id is: codex
Your assigned tasks for batch $BID are: $*

IMPORTANT -- THE STANDARD HAS CHANGED. Read the brief above carefully: you must
now score TWO INDEPENDENT axes. \`verdict\` is about IMPLEMENTATION FAITHFULNESS
ONLY. A metric that is confounded with model size is NOT thereby RED -- score
that separately under \`comparable\`. Set \`defect_location\` to CODE or CONFIG.

These tasks already have verdict files in verification_campaign/verdicts/ from
BOTH reviewers, including your own earlier ones. Read them, then CHALLENGE them.
Where the code was verified against released reference code, or reproduced a
published number, the burden is on the RED -- prior reviewers on this campaign
have been refuted by direct measurement more than once. Measure, do not defer.

Write each verdict JSON to verification_campaign/verdicts/<task>.codex.json,
overwriting the existing file, using the FULL revised schema. Do not modify any
other file."
cd ~/projects/beyond_lm_eval
codex exec --skip-git-repo-check "$PROMPT" > "$CAMP/logs/$BID.codex2.log" 2>&1
echo "codex $BID rc=$?"
