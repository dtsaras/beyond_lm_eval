#!/usr/bin/env bash
# GPT-5.6 (gpt-5.6-sol) verification pass over all paper-core tasks.
#
# Run when the codex quota resets (see CODEX_BLOCKER.md). Safe to re-run: it
# skips tasks whose codex verdict is already newer than the task source, so an
# interrupted run resumes where it stopped.
set -uo pipefail
CAMP="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$CAMP")"
cd "$ROOT" || exit 1

mapfile -t TASKS < <(python3 -c "
import json
for t in json.load(open('$CAMP/tasks.json')):
    print(t['task'] if isinstance(t, dict) else t)
")

echo "codex verification: ${#TASKS[@]} tasks"
for task in "${TASKS[@]}"; do
  out="$CAMP/verdicts/${task}.codex.json"
  src=$(python3 - "$task" <<'PY'
import sys, glob, os
t = sys.argv[1]
hits = [f for f in glob.glob('src/blme/tasks/**/*.py', recursive=True)
        if t in open(f, encoding='utf-8', errors='ignore').read()]
print(max(hits, key=os.path.getmtime) if hits else '')
PY
)
  if [[ -n "$src" && -f "$out" && "$out" -nt "$src" ]]; then
    echo "  SKIP $task (verdict newer than $src)"; continue
  fi

  echo "=== $task ==="
  timeout 3600 codex exec --skip-git-repo-check "
You are an adversarial verification reviewer for the BLME repo at $ROOT.

FIRST read $CAMP/BRIEF.md IN FULL and follow it exactly. It defines the two-axis
standard (verdict = IMPLEMENTATION FAITHFULNESS ONLY; comparability, config,
protocol and sample size go on 'comparable'/'confounds') and the AXIS LEAKAGE
rule, which quotes your own earlier self-diagnosis of that failure mode.

RUN EVERYTHING LOCALLY. Do not ssh to any host; eez128/eez129/eez130/eez131/
eez132/h5 are OFF LIMITS, eez130 especially.

Re-review the single task '$task' against the CURRENT code. The Opus 5 verdict
at $CAMP/verdicts/${task}.opus5.json records what was measured and fixed --
read it, then verify independently by MEASURING. Challenge it where the evidence
warrants: if the implementation matches the source paper's numbers or the
released reference code, say so and do not withhold GREEN for reasons that
belong on another axis. A metric that cannot be compared across models is NOT
thereby incorrect.

Note: defaults.yaml is merged INTO self.config, so code-level .get('x', DEFAULT)
fallbacks never fire.

WRITE your verdict to $CAMP/verdicts/${task}.codex.json using the exact schema
of the opus5 file (task, reviewer 'codex', verdict, verdict_reason,
defect_location, comparable, comparable_reason, normalized_variant,
normalized_ok, github, github_repo, github_reason, paper_number, paper_reason,
confounds, confounds_reason, defects, fix). Quote concrete measured numbers in
every reason field. Modify NOTHING outside that verdicts directory.
" 2>&1 | tail -5
  sleep 5
done

echo "done. regenerating the matrix:"
python3 "$CAMP/refresh.py"
