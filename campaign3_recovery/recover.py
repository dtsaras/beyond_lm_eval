#!/usr/bin/env python3
"""Reconstruct the Campaign-3 workflow return value from the crash-surviving
journal + agent transcripts (offline, zero token cost).

Journal result records carry only {agentId, key, result}. The finding->refuter
mapping the workflow computes in-memory is gone, so we rebuild it: each refuter
transcript's prompt embeds task + CLAIM title + File, which we join back to the
finder's findings.
"""
import json, glob, os, re, sys

WD = "/home/dtsaras/.claude/projects/-home-dtsaras-projects-beyond-lm-eval/86f00f68-4f87-487a-a22e-e4d6e7c1d777/subagents/workflows/wf_50bc34f1-1d7"
OUT = "/tmp/claude-1736197890/-home-dtsaras-projects-beyond-lm-eval/86f00f68-4f87-487a-a22e-e4d6e7c1d777/scratchpad/recovered_result.json"

# 1) agentId -> result, from journal (last write wins)
results = {}
for line in open(os.path.join(WD, "journal.jsonl")):
    try:
        rec = json.loads(line)
    except Exception:
        continue
    if rec.get("type") == "result" and rec.get("result") is not None:
        results[rec["agentId"]] = rec["result"]

finders = {}   # task -> finder result
reviews = {}   # dim -> review result
surveys = []   # list of {candidates:[...]}
synth = None
refuters = {}  # agentId -> {task,severity,title,file,lens,result}

REFUTE_RE = re.compile(
    r'task "(?P<task>[^"]+)"\.\s*\n'
    r'CLAIM \[(?P<sev>[^\]]+)\] (?P<title>.*?)\s*\n'
    r'File: (?P<file>[^\n]*)', re.S)


def prompt_of(agent_id):
    f = os.path.join(WD, f"agent-{agent_id}.jsonl")
    if not os.path.exists(f):
        return ""
    with open(f) as fh:
        first = fh.readline()
    try:
        rec = json.loads(first)
        c = rec["message"]["content"]
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        return c
    except Exception:
        return ""


for aid, r in results.items():
    if not isinstance(r, dict):
        continue
    if "task" in r and "parity_tests" in r:
        finders[r["task"].split(" ")[0]] = r
    elif "dimension" in r:
        reviews[r["dimension"]] = r
    elif "candidates" in r:
        surveys.append(r)
    elif "must_add" in r:
        synth = r
    elif "refuted" in r:
        p = prompt_of(aid)
        m = REFUTE_RE.search(p)
        lens = "numeric" if "REPRODUCE IT" in p else ("sources" if "PRIMARY SOURCES" in p else "?")
        if m:
            refuters[aid] = dict(task=m.group("task").split(" ")[0],
                                 severity=m.group("sev").strip(),
                                 title=m.group("title").strip(),
                                 file=m.group("file").strip(),
                                 lens=lens, result=r)
        else:
            refuters[aid] = dict(task=None, title=None, file=None, lens=lens, result=r)

# 2) index refuter votes by (task, normalized title) and (task, file)
def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())

by_title = {}
by_file = {}
for aid, rf in refuters.items():
    if rf["task"] is None:
        continue
    by_title.setdefault((rf["task"], norm(rf["title"])), []).append(rf)
    by_file.setdefault((rf["task"], norm(rf["file"])), []).append(rf)

# 3) attach votes to each finder finding -> rebuild `judged`
verify = []
match_stats = {"matched": 0, "unmatched": 0}
for task, fr in finders.items():
    judged = []
    for fd in fr.get("findings", []) or []:
        key_t = (task, norm(fd.get("title")))
        key_f = (task, norm(f"{fd.get('file','')}:{fd.get('line','')}"))
        votes_rf = by_title.get(key_t) or by_file.get(key_f) or []
        votes = [rf["result"] for rf in votes_rf]
        if votes:
            match_stats["matched"] += 1
        else:
            match_stats["unmatched"] += 1
        not_ref = [v for v in votes if not v.get("refuted")]
        judged.append(dict(fd,
                           votes=votes,
                           nvotes=len(votes),
                           survives=(len(not_ref) >= 1) if votes else True,  # unresolved -> keep
                           unresolved=(len(votes) == 0),
                           unanimous=(len(votes) > 0 and len(not_ref) == len(votes))))
    v = dict(fr)
    v["judged"] = judged
    verify.append(v)

# 4) survey block: raw candidates + synthesis (may be None)
raw = []
for s in surveys:
    raw.extend(s.get("candidates", []) or [])
survey = {"raw": raw, "synthesis": synth}

out = {"verify": verify,
       "survey": survey,
       "review": list(reviews.values()),
       "stats": {"finders": len(finders), "refuter_votes": len(refuters),
                 "reviews": len(reviews), "survey_lenses": len(surveys),
                 "synthesis": synth is not None, **match_stats}}
json.dump({"result": out, "agentCount": len(results)}, open(OUT, "w"), indent=1)
print("wrote", OUT)
print("finders:", len(finders), "| refuter votes:", len(refuters),
      "| findings matched:", match_stats["matched"], "unmatched:", match_stats["unmatched"])
print("reviews:", sorted(reviews), "| survey raw:", len(raw), "| synthesis:", synth is not None)
